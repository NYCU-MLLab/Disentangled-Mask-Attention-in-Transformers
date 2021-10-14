#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-Head Attention layer definition."""

import math
from typing import Dict, Optional, Tuple

import numpy
import torch
from torch import nn
from torch.nn import functional as F

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention

class DisentangledMaskAttention(nn.Module):
    """Multi-Head Attention layer (Disentangled attention).

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head, n_feat, dropout_rate, diesntangled=True, clusters=4, encoder_decoder_attention=False,
                 causal=False, var_estimation=False, mu_grad=5):
        """Construct an MultiHeadedAttention object."""
        super(DisentangledMaskAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)
        self.var_estimation = var_estimation

        self.clusters = clusters
        self.disentangled = diesntangled
        self.encoder_decoder_attention = encoder_decoder_attention
        self.causal = causal

        # Semantic mask attention parameters
        self.var = nn.Parameter(torch.full((1, 1, 1), 0.01), requires_grad=False)
        self.semantic_log_prior = nn.Parameter(torch.full((n_head, clusters,), math.log(1/clusters)))
        self.semantic_log_var = nn.Parameter(torch.full((n_head, clusters, n_feat//n_head), math.log(1)))
        self.semantic_mu = nn.Parameter(torch.cat([
            nn.init.normal_(torch.ones(n_head, 1, n_feat//n_head), -0.1 + 0.1/clusters*c, 0.1) for c in range(clusters)
        ], dim=1))
        self.semantic_mu.register_hook(lambda grad: mu_grad*grad)

        if var_estimation:
            self.linear_semantic_var = nn.Linear(n_feat, n_head)

            if self.encoder_decoder_attention:
                self.linear_semantic_var_k = nn.Linear(n_feat, n_head)

        # Disentangled head attention parameters
        self.head_log_prior = nn.Parameter(torch.full((1, clusters,), math.log(1/clusters)))
        self.head_log_var = nn.Parameter(torch.full((1, clusters, n_feat//n_head), math.log(1)))
        self.head_mu = nn.Parameter(torch.cat([
            nn.init.normal_(torch.ones(1, 1, n_feat//n_head), -0.1 + 0.1/clusters*c, 0.1) for c in range(clusters)
        ], dim=1))
        self.head_mu.register_hook(lambda grad: mu_grad*grad)

        if var_estimation:
            self.linear_head_var = nn.Linear(n_feat, n_head)

        self.num_updates = nn.Parameter(torch.zeros((1,)), requires_grad=False)
        self.outputs_dict: Dict[str, torch.Tensor] = {}

        self.init_params()
        self.debug = False

        if var_estimation:
            nn.init.xavier_normal_(self.linear_semantic_var.weight, gain=0.01)
            nn.init.xavier_normal_(self.linear_head_var.weight, gain=0.01)
            if self.encoder_decoder_attention:
                nn.init.xavier_normal_(self.linear_semantic_var_k.weight, gain=0.01)

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1 and p.numel() > 1:
                nn.init.xavier_normal_(p)

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

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = float(
                numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
            )
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def clustering(self, x: torch.Tensor, approx_log_var: torch.Tensor, mu: torch.Tensor,
                   log_var: torch.Tensor, log_prior: torch.Tensor,
                   mi_estimation: bool=False)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            x: [batch, heads, length, dim]
            mu: [heads, clusters, dim]
            approx_log_var: [batch, heads, length]
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
        
        x = x.float()
        approx_log_var = approx_log_var.float()
        z = self.normal_reparameterizing(x, approx_log_var.unsqueeze(-1))
        mu = mu.float()
        log_var = log_var.float()
        log_prior = log_prior.float()

        # true distribution
        z = z.unsqueeze(3) # B x H x L x 1 x D
        mu = mu.unsqueeze(1) # H x 1 x C x D
        approx_log_var = approx_log_var.unsqueeze(-1).unsqueeze(-1) # B x H x L x 1 x 1
        approx_var = approx_log_var.exp() # B x H x L x 1 x 1

        log_var = log_var.unsqueeze(1) # H x 1 x C x D        
        var = log_var.exp()
        log_prior = F.log_softmax(log_prior, dim=-1) # H x C

        mse = (z - mu).pow(2)/var # B x H x L x C x D
        log_pdf = -0.5*mse.sum(-1) - 0.5*log_var.sum(-1) - D*math.pi # B x H x L x C
        log_pdf = log_pdf + log_prior.unsqueeze(1) # B x H x L x C
        log_probs = F.log_softmax(log_pdf - log_pdf.max(dim=-1, keepdim=True).values.detach() + 5, dim=-1) # B x H x L x C
        cluster_probs = log_probs.exp()

        prior = log_prior.unsqueeze(0).unsqueeze(2).expand_as(log_probs).exp()
        kl_distribution = F.kl_div(log_probs, prior, reduction="none").sum(-1).mean((1, 2)) # B
        kl_distribution = kl_distribution + 0.5*(cluster_probs * (mse + approx_var/var + log_var).sum(-1)).sum(-1).mean((1, 2)) # B
        kl_distribution = kl_distribution - 0.5*(1 + approx_log_var).sum(-1).mul(D).mean((1, 2, 3))
        kl_loss = kl_distribution.view(-1, 1, 1)
        del mse

        div_loss = torch.einsum("bhqc,bhkc->bhqk", cluster_probs, cluster_probs)
        div_loss = (div_loss - torch.eye(L).to(x)).pow(2)
        mask = torch.eye(L) * 1.25 + (1 - torch.eye(L)) * 0.75
        mask = mask.to(x)
        div_loss = (div_loss * mask).mean((1, 2, 3))

        if mi_estimation:
            pdf = (log_pdf - log_pdf.max().detach()).exp().sum(-1, keepdim=True) # B x H x L x C
            mi_probs = F.normalize(cluster_probs*pdf, p=1., dim=2, eps=1e-6)  # B x H x L x C
            mi_probs = torch.einsum("bilc,bjlc->blij", mi_probs, cluster_probs).clamp(max=1)
            mask = 1 - torch.eye(H).to(x)
            mi_loss = -(1 - mi_probs + 1e-6).log().mul(mask).mean((1, 2, 3)) * 10
            mi_loss = mi_loss -(F.log_softmax(
                torch.einsum("bhqc,bhkc->bhqk", cluster_probs, cluster_probs), dim=-1
            ).diagonal(dim1=2, dim2=3).mean((1, 2)))
        else:
            mi_loss = torch.zeros_like(kl_loss)

        del log_pdf, log_probs, mask

        z = z.squeeze(3)

        return z, cluster_probs, kl_loss, div_loss, mi_loss

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
        H = self.h
        D_K = D//H
        L_K = key.size()[1]
        outputs = {}
        input_dtype = query.dtype

        attention_weights: Optional[torch.Tensor] = None
        if not get_attn_weights:
            attention_weights = None

        # 1. Semantic mask attention
        if self.var_estimation:
            log_var = self.linear_semantic_var(query).view(B, L_Q, H).transpose(2, 1)
        else:
            log_var = self.var.log()

        query = query.view(B, L_Q, H, D_K).transpose(2, 1)
        if self.debug:
            outputs.update({"hidden_states": query})

        query, cluster_probs_q, cluster_loss_q, cluster_div_loss_q, _ = self.clustering(
            query,
            log_var,
            self.semantic_mu,
            self.semantic_log_var,
            self.semantic_log_prior
        )
        query = query.transpose(2, 1).reshape(B, L_Q, D).to(input_dtype)

        if self.encoder_decoder_attention:
            if self.var_estimation:
                log_var = self.linear_semantic_var_k(key).view(B, L_K, H).transpose(2, 1)
            else:
                log_var = self.var.log()

            key = key.view(B, L_K, H, D_K).transpose(2, 1)
            key, cluster_probs_k, cluster_loss_k, cluster_div_loss_k, _ = self.clustering(
                key,
                log_var,
                self.semantic_mu,
                self.semantic_log_var,
                self.semantic_log_prior
            )
            key = key.transpose(2, 1).reshape(B, L_K, D).to(input_dtype)

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
        attention_mask = F.normalize((attention_mask * cluster_mask.float()), dim=-1, p=1., eps=1e-6)

        # 3. Linaer projection
        query = self.linear_q(query)
        key = self.linear_k(key).view(B, L_K, H, D_K).transpose(2, 1) # B x H x L x D
        value = self.linear_v(value).view(B, L_K, H, D_K).transpose(2, 1) # B x H x L x D

        # 4. Disentanglement head attention
        if self.var_estimation:
            log_var = self.linear_head_var(query).view(B, L_Q, H).transpose(2, 1)
        else:
            log_var = self.var.log()

        query = query.view(B, L_Q, H, D_K).transpose(2, 1)/math.sqrt(D_K) # B x H x L x D
        query, mi_cluster_probs, mi_cluster_loss, cluster_div_loss_mi, mi_loss = self.clustering(
            query,
            log_var,
            self.head_mu,
            self.head_log_var,
            self.head_log_prior,
            mi_estimation=True
        )
        query = query.to(input_dtype)

        if self.debug:
            outputs.update({"query": query})

        cluster_loss = cluster_loss + mi_cluster_loss
        cluster_div_loss = cluster_div_loss + cluster_div_loss_mi

        # 5. Softmax attention
        QK = query.matmul(key.transpose(3, 2))
        attention_weights = F.softmax(QK + padding_mask, dim=-1).float()
        mask_attention_weights = F.normalize(attention_weights*attention_mask.float(), p=1., dim=-1, eps=1e-6)
        annealing_weight = 1 - torch.maximum(0.1*torch.ones_like(self.num_updates), torch.exp(-5e-4*self.num_updates))
        attention_weights = (1 - annealing_weight) * attention_weights + annealing_weight * mask_attention_weights
        self.attn = attention_weights

        # 6. Attention output
        attn_output = torch.einsum("bhqk,bhkd->bhqd", attention_weights.type_as(value), value)
        attn_output = attn_output.view(B, H, L_Q, D_K).transpose(2, 1).contiguous().view(B, -1, D)
        attn_output = self.linear_out(attn_output)

        del QK

        outputs.update({
            "loss_kl": cluster_loss.view(-1, 1, 1),
            "loss_mi": mi_loss.view(-1, 1, 1),
            "loss_div": cluster_div_loss.view(-1, 1, 1)
        })

        if self.debug:
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

        self.outputs_dict = outputs

        return attn_output

    def forward_full(self, query, key, value, mask):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        if not self.disentangled:
            return self.forward_full(query, key, value, mask)
        else:
            if mask is None:
                padding_mask = torch.zeros((key.size()[0], 1, key.size()[1])).to(query)
            else:
                padding_mask = 1 - mask.to(query)

            return self.disentangled_attention(query, key, value, padding_mask)

class LegacyRelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (old version).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.

    """

    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, time2).

        Returns:
            torch.Tensor: Output tensor.

        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor (#batch, time1, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (new implementation).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.

    """

    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, 2*time1-1).
            time1 means the length of query vector.

        Returns:
            torch.Tensor: Output tensor.

        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[
            :, :, :, : x.size(-1) // 2 + 1
        ]  # only keep the positions from 0 to time2

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, 2*time1-1, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, 2*time1-1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)
