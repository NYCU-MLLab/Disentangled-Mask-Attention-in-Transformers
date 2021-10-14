import math
from typing import Optional, Tuple, Dict

import torch
from torch import nn
from torch.nn import functional as F

class DisentangledAttention(nn.Module):
    """
    Diesntangled attention.
    """
    def __init__(self, heads: int, dims: int, clusters: int, causal: bool=False,
                 self_attention: bool= True, encoder_decoder_attention: bool= False,
                 grad_multiplier: float= 10.):
        super(DisentangledAttention, self).__init__()

        self.heads = heads
        self.dims = dims
        self.clusters = clusters
        self.causal = causal
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        self.eps = 1e-6
        self.grad_multiplier = grad_multiplier

        # Attention parameters
        self.linear_query = nn.Linear(dims, dims)
        self.linear_key = nn.Linear(dims, dims)
        self.linear_value = nn.Linear(dims, dims)
        self.linear_output = nn.Linear(dims, dims)

        # Semantic mask attention parameters
        self.var = nn.Parameter(torch.full((1, 1, 1), 0.01), requires_grad=False)
        self.tok_log_prior = nn.Parameter(torch.full((heads, clusters,), math.log(1/clusters)))
        self.tok_log_var = nn.Parameter(torch.full((heads, clusters, dims//heads), math.log(1)))
        self.tok_mu = nn.Parameter(torch.cat([
            nn.init.normal_(torch.ones(heads, 1, dims//heads), -0.1 + 1/clusters*c, 0.1) for c in range(clusters)
        ], dim=1))
        self.tok_mu.register_hook(lambda grad: grad_multiplier*grad)

        # Disentangled head attention parameters
        self.head_log_prior = nn.Parameter(torch.full((1, clusters,), math.log(1/clusters)))
        self.head_log_var = nn.Parameter(torch.full((1, clusters, dims//heads), math.log(1)))
        self.head_mu = nn.Parameter(torch.cat([
            nn.init.normal_(torch.ones(1, 1, dims//heads), -0.1 + 1/clusters*c, 0.1) for c in range(clusters)
        ], dim=1))
        self.head_mu.register_hook(lambda grad: grad_multiplier*grad)

        self.num_updates = nn.Parameter(torch.zeros((1,)), requires_grad=False)
        self.apply(self.init_params)
        self.init_cluster_params()

        self.outputs_dict = {}
        self.debug = False

    def init_params(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1/math.sqrt(2))

    def init_cluster_params(self):
        params = [
            self.tok_log_prior,
            self.tok_log_var,
            self.tok_mu,
            self.head_log_prior,
            self.head_log_var,
            self.head_mu,
        ]

        for p in params:
            nn.init.xavier_normal_(p, gain=1)

    def normal_reparameterizing(self, x: torch.Tensor, log_var: torch.Tensor)-> torch.Tensor:
        """
        Parameters:
            x

        Return:
            same shape as x
        """
        if self.training:
            x = x + (log_var/2.).exp()*torch.randn_like(x)

        return x

    def clustering(self, x: torch.Tensor, approx_log_var: torch.Tensor, mu: torch.Tensor,
                   log_var: torch.Tensor, log_prior: torch.Tensor,
                   mi_estimation: bool=False)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            x: [batch, heads, length, dim]
            approx_log_var: [batch, heads, length]
            mu: [heads, clusters, dim]
            log_var: [heads, clsuters, dim]
            log_prior: [heads, clusters]

        Returns:
            (probs, kl_loss, div_loss, mi_loss)
                probs: [batch, length, clusters]
                kl_loss: [batch,]
                div_loss: [batch,]
                mi_loss: [batch,]
        """
        B, H, L, D = x.size()

        # true distribution
        x = x.unsqueeze(3) # B x H x L x 1 x D
        mu = mu.unsqueeze(1) # H x 1 x C x D
        approx_log_var = approx_log_var.unsqueeze(-1).unsqueeze(-1) # B x H x L x 1 x 1
        approx_var = approx_log_var.exp() # B x H x L x 1 x 1

        log_var = log_var.unsqueeze(1).float() # H x 1 x C x D        
        var = log_var.exp()
        log_prior = F.log_softmax(log_prior, dim=-1) # H x C

        mse = (x - mu).pow(2)/var # B x H x L x C x D
        log_pdf = -0.5*mse.sum(-1) - D*math.pi - 0.5*log_var.sum(-1) # B x H x L x C
        log_pdf = log_pdf + log_prior.unsqueeze(1) # B x H x L x C
        log_probs = F.log_softmax(log_pdf, dim=-1) # B x H x L x C
        cluster_probs = log_probs.exp().float()

        prior = log_prior.unsqueeze(0).unsqueeze(2).expand_as(log_probs).exp().float()
        kl_distribution = F.kl_div(log_probs, prior, reduction="none").sum(-1).mean((1, 2)).float() # B
        kl_distribution = kl_distribution + 0.5*(cluster_probs * (mse + approx_var/var + log_var).sum(-1)).sum(-1).mean((1, 2)) # B
        kl_distribution = kl_distribution - 0.5*(1 + approx_log_var).sum(-1).mul(D).mean((1, 2, 3))
        kl_loss = kl_distribution.view(-1, 1, 1)

        div_loss = torch.einsum("bhqc,bhkc->bhqk", cluster_probs, cluster_probs)
        div_loss = (div_loss - torch.eye(L).to(x)).pow(2)
        mask = torch.eye(L) * 1.25 + (1 - torch.eye(L)) * 0.75
        mask = mask.to(x)
        div_loss = (div_loss * mask).mean((1, 2, 3))

        if mi_estimation:
            pdf = (log_pdf - log_pdf.max().detach()).float().exp().sum(-1, keepdim=True) # B x H x L x C
            mi_probs = F.normalize(cluster_probs*pdf, p=1., dim=2, eps=1e-6)  # B x H x L x C
            mi_probs = torch.einsum("bilc,bjlc->blij", mi_probs, cluster_probs).clamp(max=1)
            mask = 1 - torch.eye(H).to(x)
            mi_loss = -(1 - mi_probs + 1e-6).log().mul(mask).mean((1, 2, 3)) * 10            
            mi_loss = mi_loss - (F.log_softmax(
                torch.einsum("bhqc,bhkc->bhqk", cluster_probs, cluster_probs).float(), dim=-1
            ).diagonal(dim1=2, dim2=3).mean((1, 2)))
        else:
            mi_loss = torch.zeros_like(kl_loss)

        return cluster_probs, kl_loss, div_loss, mi_loss

    def softmax_attention(self, query:torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                          padding_mask: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Shape:
            query/key/value: [batch, heads, length, dims]
            padding_mask: [batch, heads, length q, legnth k] or [batch, heads, 1, legnth k]
            weights: [batch, heads, length k, 1]
        """
        raw_attn_weights = torch.matmul(query, key.transpose(3, 2)) + padding_mask.detach()*-10000        
        attn_weights = F.softmax(raw_attn_weights, -1) * (1 - padding_mask)
        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights, raw_attn_weights

    def full_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                      padding_mask: torch.Tensor, get_attn_weights: bool=True, debug: bool=False)-> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Parameters:
            query: Query tensor. [batch, heads, length q, dims]
            key: Key tensor. [batch, heads, length k, dims]
            value: Value tensor. [batch, heads, length v, dims]
            padding_mask: Padding mask of input tensor. [batch, heads, length q, length k] or [batch, 1, length k]
            weights: [batch, heads, length k, 1]
            get_attn_weights: Get full attention weights. (Consume more memory)

        Returns:(attn_output, attn_weights)
                attn_output: [batch, heads, length q, dims]
                attn_weights: [batch, heads, legnth q, length k]
        """     
        B, L_Q, D = query.size()
        L_K = key.size()[1]
        L_V = value.size()[1]
        H = self.heads
        D_K = D//H

        query = self.linear_query(query)
        key = self.linear_key(key)
        value = self.linear_value(value)

        query = query.view(B, L_Q, H, D_K).transpose(2, 1)
        key = key.view(B, L_K, H, D_K).transpose(2, 1)
        value = value.view(B, L_V, H, D_K).transpose(2, 1)

        query = query/math.sqrt(D_K)

        if self.causal and L_Q == L_K:
            padding_mask = 1 - torch.tril(torch.ones((L_Q, L_K))).to(query)

        attn_output, attn_weights, raw_attn_weights = self.softmax_attention(query, key, value, padding_mask)
        attn_output = attn_output.transpose(2, 1).contiguous().view(B, L_Q, D)
        attn_output = self.linear_output(attn_output)

        outputs = {}
        if debug or self.debug:
            outputs.update({
                "query": query
            })
            outputs.update({
                "mask_attn": attn_weights
            })

        attn_weights: Optional[torch.Tensor] = attn_weights
        if not get_attn_weights:
            attn_weights = None

        return attn_output, attn_weights, raw_attn_weights, outputs

    def disentangled_attention(self, query: torch.Tensor, key: torch.Tensor, value:torch.Tensor, 
                               padding_mask: torch.Tensor, get_attn_weights: bool=True,
                               debug: bool=False)-> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Parameters:
            query: Query tensor. [batch, length q, dims]
            key: Key tensor. [batch, length k, dims]
            value: Value tensor. [batch, length v, dims]
            padding_mask: Padding mask of input tensor. [batch, 1, length k]
            get_attn_weights: Get full attention weights. (Consume more memory)
            debug

        Returns:(attn_output, attn_weights, outputs)
                attn_output: [batch, length q, dims]
                attn_weights: [batch, legnth q, length k]
                outputs: {
                    "cluster_loss": [batch, 1, 1],
                    "cluster_div_loss": [batch, 1, 1],
                    "mi_loss": [batch, 1, 1],
                    "annealing_weight": [batch, 1, 1],

                    "cluster_probs_q": [batch, heads, length q, clusters] if debug = True,
                    "cluster_probs_k": [batch, heads, length k, clusters] if debug = True,
                    "mi_cluster_probs": [batch, heads, length q, clusters] if debug = True,
                }
        """     
        B, L_Q, D = query.size()
        H = self.heads
        D_K = D//H
        L_K = key.size()[1]
        outputs = {}

        attention_weights: Optional[torch.Tensor] = None
        if not get_attn_weights:
            attention_weights = None

        # 1. Semantic mask attention
        query = query.view(B, L_Q, H, D_K).transpose(2, 1)

        if self.grad_multiplier > 1:
            query = self.normal_reparameterizing(query, self.var.log().unsqueeze(-1))
        else:
            query = query/10.

        cluster_probs_q, cluster_loss_q, cluster_div_loss_q, _ = self.clustering(
            query,
            self.var.log(),
            self.tok_mu,
            self.tok_log_var,
            self.tok_log_prior
        )
        query = query.transpose(2, 1).reshape(B, L_Q, D)

        if self.grad_multiplier == 1:
            query = self.normal_reparameterizing(query*10, self.var.log().unsqueeze(-1))

        if debug or self.debug:
            outputs.update({
                "hidden_states": query.view(B, L_Q, H, D_K).transpose(2, 1)
            })

        if self.encoder_decoder_attention:
            key = key.view(B, L_K, H, D_K).transpose(2, 1)

            if self.grad_multiplier > 1:
                key = self.normal_reparameterizing(key, self.var.log().unsqueeze(-1))
            else:
                key = key / 10.

            cluster_probs_k, cluster_loss_k, cluster_div_loss_k, _ = self.clustering(
                key,
                self.var.log(),
                self.tok_mu,
                self.tok_log_var,
                self.tok_log_prior
            )
            key = key.transpose(2, 1).reshape(B, L_K, D)

            if self.grad_multiplier == 1:
                key = self.normal_reparameterizing(key*10., self.var.log().unsqueeze(-1))

            cluster_loss = cluster_loss_q + cluster_loss_k
            cluster_div_loss = cluster_div_loss_q + cluster_div_loss_k
        else:
            cluster_probs_k = cluster_probs_q
            cluster_loss = cluster_loss_q
            cluster_div_loss = cluster_div_loss_q

        # 2. Attention mask
        if not self.encoder_decoder_attention:
            cluster_mask = torch.einsum("bhqc,bhkc->bhqk", cluster_probs_q, cluster_probs_q)
        else:
            cluster_mask = torch.einsum("bhqc,bhkc->bhqk", cluster_probs_q, cluster_probs_k)

        if self.causal and L_Q == L_K:
            attention_mask = torch.tril(torch.ones((L_Q, L_K))).to(query)
        else:                
            attention_mask = 1 - padding_mask.unsqueeze(1) # B x 1 x 1 x L

        padding_mask = torch.zeros_like(attention_mask).masked_fill_(attention_mask == 0, float("-inf"))
        attention_mask = attention_mask * cluster_mask
        attention_mask = F.normalize(attention_mask.float(), dim=-1, p=1., eps=1e-6)
        annealing_weight = 1 - torch.maximum(0.1*torch.ones_like(self.num_updates), torch.exp(-5e-4*self.num_updates))

        # 3. Linaer projection
        query = self.linear_query(query) # B x H x L x D
        key = self.linear_key(key).view(B, L_K, H, D_K).transpose(2, 1) # B x H x L x D
        value = self.linear_value(value).view(B, L_K, H, D_K).transpose(2, 1) # B x H x L x D

        # 4. Disentanglement head attention
        query = query.view(B, L_Q, H, D_K).transpose(2, 1)/math.sqrt(D_K)

        if self.grad_multiplier > 1:
            query = self.normal_reparameterizing(query, self.var.log().unsqueeze(-1))
        else:
            query = query / 10.

        mi_cluster_probs, mi_cluster_loss, cluster_div_loss_mi, mi_loss = self.clustering(
            query,
            self.var.log(),
            self.head_mu,
            self.head_log_var,
            self.head_log_prior,
            mi_estimation=True
        )

        if self.grad_multiplier == 1:
            query = self.normal_reparameterizing(query*10., self.var.log().unsqueeze(-1))

        if debug or self.debug:
            outputs.update({
                "query": query
            })

        cluster_loss = cluster_loss + mi_cluster_loss
        cluster_div_loss = cluster_div_loss + cluster_div_loss_mi

        # 5. Softmax attention
        QK = query.matmul(key.transpose(3, 2))
        attention_weights = F.softmax(QK + padding_mask, dim=-1).float()

        if debug or self.debug:
            _attention_weights = attention_weights

            # accumulate for auto-regressive inference
            if self.causal or self.encoder_decoder_attention:
                if "attn" in self.outputs_dict:
                    if self.causal:
                        self.outputs_dict["attn"] = F.pad(
                            self.outputs_dict["attn"], [0, 1],
                            value=0.
                        )

                    _attention_weights = torch.cat([
                        self.outputs_dict["attn"], attention_weights
                    ], dim=-2)

            outputs.update({
                "attn": _attention_weights
            })

        mask_attention_weights = F.normalize(attention_weights*attention_mask.float(), p=1., dim=-1, eps=1e-6)
        attention_weights = (1 - annealing_weight) * attention_weights + annealing_weight * mask_attention_weights

        if debug or self.debug:
            _attention_mask = attention_mask
            _attention_weights = attention_weights

            # accumulate for auto-regressive inference
            if self.causal or self.encoder_decoder_attention:
                if L_Q == 1 and L_K > 1:
                    if "mask" in self.outputs_dict:
                        if self.causal:
                            self.outputs_dict["mask"] = F.pad(
                                self.outputs_dict["mask"], [0, 1],
                                value=0.
                            )
                            self.outputs_dict["mask_attn"] = F.pad(
                                self.outputs_dict["mask_attn"], [0, 1],
                                value=0.
                            )

                        _attention_mask = torch.cat([
                            self.outputs_dict["mask"], attention_mask
                        ], dim=-2)
                        _attention_weights = torch.cat([
                            self.outputs_dict["mask_attn"], attention_weights
                        ], dim=-2)

            outputs.update({
                "mask": _attention_mask,
                "mask_attn": _attention_weights
            })

        # 6. Attention output
        attn_output = torch.einsum("bhqk,bhkd->bhqd", attention_weights.type_as(value), value)
        attn_output = attn_output.view(B, H, L_Q, D_K).transpose(2, 1).contiguous().view(B, -1, D)
        attn_output = self.linear_output(attn_output)

        del QK, mask_attention_weights

        outputs.update({
            "cluster_loss": cluster_loss.view(-1, 1, 1),
            "mi_loss": mi_loss.view(-1, 1, 1),
            "cluster_div_loss": cluster_div_loss.view(-1, 1, 1),
            "annealing_weight": annealing_weight.repeat(B).view(-1, 1, 1)
        })

        if debug or self.debug:
            outputs.update({
                "cluster_probs_q": cluster_probs_q,
                "mi_cluster_probs": mi_cluster_probs,
            })

            if self.encoder_decoder_attention:
                outputs.update({
                    "cluster_probs_k": cluster_probs_k,
                })
            else:
                outputs.update({
                    "cluster_probs_k": cluster_probs_q,
                })

        return attn_output, attention_weights, outputs

    def forward(self, query: torch.Tensor, key: torch.Tensor, value:torch.Tensor,
                padding_mask: Optional[torch.Tensor]=None, regular_attn=False, get_attn_weights: bool=True,
                debug: bool=False)-> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Parameters:
            query: Query tensor. [batch, length q, dims]
            key: Key tensor. [batch, length k, dims]
            value: Value tensor. [batch, length v, dims]
            padding_mask: Padding mask of input tensor. [batch, length q, length k] or [batch, 1, length k]
            get_attn_weights: Get full attention weights. (Consume more memory)
            regular_attn
            tau: Parameter of gumbel softmax.

        Returns: (attn_output, attn_weights, outputs_dict)
                 attn_output: [batch, length q, dims]
                 attn_weights: [batch, heads, legnth q, length k] or None
                 outputs_dict: {
                    "cluster_loss": [batch, 1, 1],
                    "cluster_div_loss": [batch, 1, 1],
                    "mi_loss": [batch, 1, 1],
                    "annealing_weight": [batch, 1, 1],

                    "cluster_probs_q": [batch, heads, length q, clusters] if debug = True,
                    "cluster_probs_k": [batch, heads, length k, clusters] if debug = True,
                    "mi_cluster_probs": [batch, heads, length q, clusters] if debug = True,                
                }
        """
        B, L_Q, D = query.size()
        L_K = key.size()[1]

        if padding_mask is None:
            if not regular_attn:
                padding_mask = torch.zeros((B, 1, L_K), device=query.device, dtype=query.dtype)
            else:
                if self.causal:
                    if L_Q == L_K:
                        padding_mask = 1 - torch.tril(torch.ones((L_Q, L_K), device=query.device, dtype=query.dtype))
                    else:
                        padding_mask = torch.zeros((L_Q, L_K), device=query.device, dtype=query.dtype)
                    padding_mask = padding_mask.unsqueeze(0)
                else:
                    padding_mask = torch.zeros((B, 1, L_K), device=query.device, dtype=query.dtype)

        if not regular_attn:
            attn_output, attn_weights, outputs_dict = self.disentangled_attention(
                query, key, value, padding_mask, get_attn_weights, debug
            )
        else:
            attn_output, attn_weights, _, outputs_dict = self.full_attention(
                query, key, value, padding_mask.unsqueeze(1), get_attn_weights, debug
            )

        self.outputs_dict = outputs_dict

        return attn_output, attn_weights, outputs_dict
