"""Library implementing attention modules.

Authors
 * Ju-Chieh Chou 2020
 * Jianyuan Zhong 2020
 * Loren Lugosch 2020
 * Samuele Cornell 2020
"""

import torch
import logging
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple

from torch.nn.modules.activation import GLU
from speechbrain.dataio.dataio import length_to_mask
import torch.nn.functional as F
import math


logger = logging.getLogger(__name__)


class ContentBasedAttention(nn.Module):
    """ This class implements content-based attention module for seq2seq
    learning.

    Reference: NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN
    AND TRANSLATE, Bahdanau et.al. https://arxiv.org/pdf/1409.0473.pdf

    Arguments
    ---------
    attn_dim : int
        Size of the attention feature.
    output_dim : int
        Size of the output context vector.
    scaling : float
        The factor controls the sharpening degree (default: 1.0).

    Example
    -------
    >>> enc_tensor = torch.rand([4, 10, 20])
    >>> enc_len = torch.ones([4]) * 10
    >>> dec_tensor = torch.rand([4, 25])
    >>> net = ContentBasedAttention(enc_dim=20, dec_dim=25, attn_dim=30, output_dim=5)
    >>> out_tensor, out_weight = net(enc_tensor, enc_len, dec_tensor)
    >>> out_tensor.shape
    torch.Size([4, 5])
    """

    def __init__(self, enc_dim, dec_dim, attn_dim, output_dim, scaling=1.0):
        super(ContentBasedAttention, self).__init__()

        self.mlp_enc = nn.Linear(enc_dim, attn_dim)
        self.mlp_dec = nn.Linear(dec_dim, attn_dim)
        self.mlp_attn = nn.Linear(attn_dim, 1, bias=False)
        self.mlp_out = nn.Linear(enc_dim, output_dim)

        self.scaling = scaling

        self.softmax = nn.Softmax(dim=-1)

        # reset the encoder states, lengths and masks
        self.reset()

    def reset(self):
        """Reset the memory in the attention module.
        """
        self.enc_len = None
        self.precomputed_enc_h = None
        self.mask = None

    def forward(self, enc_states, enc_len, dec_states):
        """Returns the output of the attention module.

        Arguments
        ---------
        enc_states : torch.Tensor
            The tensor to be attended.
        enc_len : torch.Tensor
            The real length (without padding) of enc_states for each sentence.
        dec_states : torch.Tensor
            The query tensor.

        """

        if self.precomputed_enc_h is None:

            self.precomputed_enc_h = self.mlp_enc(enc_states)
            self.mask = length_to_mask(
                enc_len, max_len=enc_states.size(1), device=enc_states.device
            )

        dec_h = self.mlp_dec(dec_states.unsqueeze(1))
        attn = self.mlp_attn(
            torch.tanh(self.precomputed_enc_h + dec_h)
        ).squeeze(-1)

        # mask the padded frames
        attn = attn.masked_fill(self.mask == 0, -np.inf)
        attn = self.softmax(attn * self.scaling)

        # compute context vectors
        # [B, 1, L] X [B, L, F]
        context = torch.bmm(attn.unsqueeze(1), enc_states).squeeze(1)
        context = self.mlp_out(context)

        return context, attn


class LocationAwareAttention(nn.Module):
    """This class implements location-aware attention module for seq2seq learning.

    Reference: Attention-Based Models for Speech Recognition, Chorowski et.al.
    https://arxiv.org/pdf/1506.07503.pdf

    Arguments
    ---------
    attn_dim : int
        Size of the attention feature.
    output_dim : int
        Size of the output context vector.
    conv_channels : int
        Number of channel for location feature.
    kernel_size : int
        Kernel size of convolutional layer for location feature.
    scaling : float
        The factor controls the sharpening degree (default: 1.0).

    Example
    -------
    >>> enc_tensor = torch.rand([4, 10, 20])
    >>> enc_len = torch.ones([4]) * 10
    >>> dec_tensor = torch.rand([4, 25])
    >>> net = LocationAwareAttention(
    ...     enc_dim=20,
    ...     dec_dim=25,
    ...     attn_dim=30,
    ...     output_dim=5,
    ...     conv_channels=10,
    ...     kernel_size=100)
    >>> out_tensor, out_weight = net(enc_tensor, enc_len, dec_tensor)
    >>> out_tensor.shape
    torch.Size([4, 5])
    """

    precomputed_enc_h: Optional[torch.Tensor]

    def __init__(
        self,
        enc_dim,
        dec_dim,
        attn_dim,
        output_dim,
        conv_channels,
        kernel_size,
        scaling=1.0,
    ):
        super(LocationAwareAttention, self).__init__()

        self.mlp_enc = nn.Linear(enc_dim, attn_dim)
        self.mlp_dec = nn.Linear(dec_dim, attn_dim)
        self.mlp_attn = nn.Linear(attn_dim, 1, bias=False)
        self.conv_loc = nn.Conv1d(
            1,
            conv_channels,
            kernel_size=2 * kernel_size + 1,
            padding=kernel_size,
            bias=False,
        )
        self.mlp_loc = nn.Linear(conv_channels, attn_dim)
        self.mlp_attn = nn.Linear(attn_dim, 1, bias=False)
        self.mlp_out = nn.Linear(enc_dim, output_dim)

        self.scaling = scaling

        self.softmax = nn.Softmax(dim=-1)

        # reset the encoder states, lengths and masks
        self.reset()

    def reset(self):
        """Reset the memory in attention module.
        """
        self.enc_len = None
        self.precomputed_enc_h = None
        self.mask = None
        self.prev_attn = None

    def forward(self, enc_states, enc_len, dec_states):
        """Returns the output of the attention module.

        Arguments
        ---------
        enc_states : torch.Tensor
            The tensor to be attended.
        enc_len : torch.Tensor
            The real length (without padding) of enc_states for each sentence.
        dec_states : torch.Tensor
            The query tensor.
        """
        if self.precomputed_enc_h is None:

            self.precomputed_enc_h = self.mlp_enc(enc_states)
            self.mask = length_to_mask(
                enc_len, max_len=enc_states.size(1), device=enc_states.device
            )

            # multiply mask by 1/Ln for each row
            self.prev_attn = self.mask * (1 / enc_len.float()).unsqueeze(1)

        # compute location-aware features
        # [B, 1, L] -> [B, C, L]
        attn_conv = self.conv_loc(self.prev_attn.unsqueeze(1))
        # [B, C, L] -> [B, L, C] -> [B, L, F]
        attn_conv = self.mlp_loc(attn_conv.transpose(1, 2))

        dec_h = self.mlp_dec(dec_states.unsqueeze(1))
        attn = self.mlp_attn(
            torch.tanh(self.precomputed_enc_h + dec_h + attn_conv)
        ).squeeze(-1)

        # mask the padded frames
        attn = attn.masked_fill(self.mask == 0, -np.inf)
        attn = self.softmax(attn * self.scaling)

        # set prev_attn to current attn for the next timestep
        self.prev_attn = attn.detach()

        # compute context vectors
        # [B, 1, L] X [B, L, F]
        context = torch.bmm(attn.unsqueeze(1), enc_states).squeeze(1)
        context = self.mlp_out(context)

        return context, attn


class KeyValueAttention(nn.Module):
    """ This class implements a single-headed key-value attention module for seq2seq
    learning.

    Reference: "Attention Is All You Need" by Vaswani et al., sec. 3.2.1

    Arguments
    ---------
    enc_dim : int
        Size of the encoder feature vectors from which keys and values are computed.
    dec_dim : int
        Size of the decoder feature vectors from which queries are computed.
    attn_dim : int
        Size of the attention feature.
    output_dim : int
        Size of the output context vector.

    Example
    -------
    >>> enc_tensor = torch.rand([4, 10, 20])
    >>> enc_len = torch.ones([4]) * 10
    >>> dec_tensor = torch.rand([4, 25])
    >>> net = KeyValueAttention(enc_dim=20, dec_dim=25, attn_dim=30, output_dim=5)
    >>> out_tensor, out_weight = net(enc_tensor, enc_len, dec_tensor)
    >>> out_tensor.shape
    torch.Size([4, 5])
    """

    def __init__(self, enc_dim, dec_dim, attn_dim, output_dim):
        super(KeyValueAttention, self).__init__()

        self.key_linear = nn.Linear(enc_dim, attn_dim)
        self.query_linear = nn.Linear(dec_dim, attn_dim)
        self.value_linear = nn.Linear(enc_dim, output_dim)
        self.scaling = torch.sqrt(torch.tensor(attn_dim).float())

        # reset the encoder states, lengths and masks
        self.reset()

    def reset(self):
        """Reset the memory in the attention module.
        """
        self.values = None
        self.keys = None
        self.mask = None

    def forward(self, enc_states, enc_len, dec_states):
        """Returns the output of the attention module.

        Arguments
        ---------
        enc_states : torch.Tensor
            The tensor to be attended.
        enc_len : torch.Tensor
            The real length (without padding) of enc_states for each sentence.
        dec_states : torch.Tensor
            The query tensor.
        """

        if self.keys is None:

            self.keys = self.key_linear(enc_states)
            self.values = self.value_linear(enc_states)
            self.mask = length_to_mask(
                enc_len, max_len=enc_states.size(1), device=enc_states.device
            ).unsqueeze(2)

        query = self.query_linear(dec_states).unsqueeze(2)
        scores = torch.matmul(self.keys, query) / self.scaling
        scores = scores.masked_fill(self.mask == 0, -np.inf)
        normalized_scores = scores.softmax(1).transpose(1, 2)
        out = torch.matmul(normalized_scores, self.values).squeeze(1)
        return out, normalized_scores


class RelPosEncXL(nn.Module):
    """

    """

    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim

        inv_freq = torch.exp(
            torch.arange(0, self.emb_dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.emb_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor
        input tensor with shape seq_len, batch_size, embed_dim
        Returns
        -------
        pos_emb : torch.Tensor
        """
        seq_len = x.size(1)
        with torch.no_grad():
            tot_pe = torch.zeros((2, seq_len, self.emb_dim), dtype=x.dtype).to(
                x
            )
            pe_past = tot_pe[0]
            pe_future = tot_pe[1]
            positions = (
                torch.arange(0, seq_len, dtype=x.dtype).to(x).unsqueeze(-1)
            )
            sinusoids = torch.sin(positions * self.inv_freq)
            pe_past[:, 0::2] = sinusoids
            pe_past[:, 1::2] = torch.cos(positions * self.inv_freq)
            pe_future[:, 0::2] = sinusoids  # same for past and future
            pe_future[:, 1::2] = torch.cos(-positions * self.inv_freq)

            pe_past = torch.flip(pe_past, (0,)).unsqueeze(0)
            pe_future = pe_future[1:].unsqueeze(0)
            pe = torch.cat([pe_past, pe_future], dim=1)
            # pe is now 1, 2*seq_len, embed_dim
            return pe


class RelPosMHAXL(nn.Module):
    """ This class implements the relative multihead implementation similar to that in Transformer XL
    https://arxiv.org/pdf/1901.02860.pdf

    Arguments
    ---------
    embed_dim : int
        Size of the encoder feature vectors from which keys and values are computed.
    num_heads: int
        Number of attention heads.
    dropout : float, optional
        Dropout rate.
    vbias: bool, optional
        Whether to use bias for computing value.
    vdim: int, optional
        Size for value. Default is embed_dim (Note each head is embed_dim // num_heads).
    mask_pos_future: bool, optional
        Whether to mask future positional encodings values.
        Must be true for causal applications e.g. decoder.
    Example
    -------
    >>> inputs = torch.rand([6, 60, 512])
    >>> pos_emb = torch.rand([1, 2*60-1, 512])
    >>> net = RelPosMHAXL(num_heads=8, embed_dim=inputs.shape[-1])
    >>> outputs, attn = net(inputs, inputs, inputs, pos_emb)
    >>> outputs.shape
    torch.Size([6, 60, 512])
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        vbias=False,
        vdim=None,
        mask_pos_future=False,
    ):
        super(RelPosMHAXL, self).__init__()
        self.embed_dim = embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.vdim == embed_dim
        self.mask_pos_future = mask_pos_future
        self.vbias = vbias

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.vhead_dim = self.vdim // num_heads

        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        assert (
            self.vhead_dim * num_heads == self.vdim
        ), "vdim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.qk_proj_weight = nn.Parameter(
                torch.empty(2 * embed_dim, embed_dim)
            )
            self.v_proj_weight = nn.Parameter(torch.empty(self.vdim, embed_dim))
        else:
            self.in_proj_weight = nn.Parameter(
                torch.empty(3 * embed_dim, embed_dim)
            )

        if vbias:
            self.value_bias_weight = nn.Parameter(torch.empty(self.vdim))
        else:
            self.vbias = None

        self.dropout_att = nn.Dropout(dropout)
        self.out_proj = nn.Linear(self.vdim, embed_dim)

        self.linear_pos = nn.Linear(embed_dim, embed_dim, bias=False)

        self.pos_bias_u = nn.Parameter(
            torch.empty(self.head_dim, self.num_heads)
        )
        self.pos_bias_v = nn.Parameter(
            torch.empty(self.head_dim, self.num_heads)
        )

        if next(self.parameters()).dtype == torch.float16:
            self.attn_fill_value = -65000
        else:
            self.attn_fill_value = -float("inf")

        self._reset_parameters()
        self.scale = 1 / math.sqrt(self.embed_dim)

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            torch.nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            torch.nn.init.xavier_uniform_(self.qk_proj_weight)
            torch.nn.init.xavier_uniform_(self.v_proj_weight)

        if self.vbias is not None:
            torch.nn.init.constant_(self.value_bias_weight, 0.0)

        # positional biases
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        # batch, head, time1, 2*time1-1.

        zero_pad = torch.zeros(
            (*x.size()[:3], 1), device=x.device, dtype=x.dtype
        )
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[
            :, :, :, : x.size(-1) // 2 + 1
        ]  # only keep the positions from 0 to time2

        if self.mask_pos_future:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(
        self,
        query,
        key,
        value,
        pos_embs,
        key_padding_mask=None,
        attn_mask=None,
        return_attn_weights=True,
    ):
        """
        Arguments
        ----------
        query : tensor
            (B, L, E) where L is the target sequence length,
            B is the batch size, E is the embedding dimension.
        key : tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        value : tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        pos_emb : tensor
            bidirectional sinusoidal positional embedding tensor (1, 2*S-1, E) where S is the max length between source and target sequence lengths,
            and E is the embedding dimension.
        key_padding_mask : tensor
            (B, S) where B is the batch size, S is the source sequence
            length. If a ByteTensor is provided, the non-zero positions will
            be ignored while the position with the zero positions will be
            unchanged. If a BoolTensor is provided, the positions with the
            value of True will be ignored while the position with the value
            of False will be unchanged.
        attn_mask : tensor
            2D mask (L, S) where L is the target sequence length, S is
            the source sequence length.
            3D mask (N*num_heads, L, S) where N is the batch
            size, L is the target sequence length, S is the source sequence
            length. attn_mask ensure that position i is allowed to attend the
            unmasked positions. If a ByteTensor is provided, the non-zero
            positions are not allowed to attend while the zero positions will
            be unchanged. If a BoolTensor is provided, positions with True is
            not allowed to attend while False values will be unchanged. If a
            FloatTensor is provided, it will be added to the attention weight.

        Outputs
        -------
        out : tensor
            (B, L, E) where L is the target sequence length, B is the
            batch size, E is the embedding dimension.
        attn_score : tensor
            (B, L, S) where B is the batch size, L is the target
            sequence length, S is the source sequence length.
        """

        # query, key and value are of shape batch, time, embed_dim
        bsz = query.shape[0]
        klen = key.shape[1]
        qlen = query.shape[1]

        if self._qkv_same_embed_dim:
            # self-attention
            if (query is key or torch.equal(query, key)) and (
                key is value or torch.equal(key, value)
            ):
                query, key, value = (
                    nn.functional.linear(query, self.in_proj_weight)
                    .view(bsz, -1, self.num_heads, self.head_dim * 3)
                    .chunk(3, dim=-1)
                )
            else:
                qweight, kweight, vweight = self.in_proj_weight.chunk(3, dim=0)
                query = nn.functional.linear(query, qweight).view(
                    bsz, -1, self.num_heads, self.head_dim
                )
                key = nn.functional.linear(key, kweight).view(
                    bsz, -1, self.num_heads, self.head_dim
                )
                value = nn.functional.linear(value, vweight).view(
                    bsz, -1, self.num_heads, self.head_dim
                )
        else:
            raise NotImplementedError
            query, key = (
                nn.functional.linear(query, self.qk_proj_weight)
                .view(bsz, -1, self.num_heads, self.head_dim * 2)
                .chunk(2, dim=-1)
            )
            value = nn.functional.linear(value, self.v_proj_weight).view(
                bsz, -1, self.num_heads, self.vhead_dim
            )

        if self.vbias is not None:
            value = value + self.value_bias_weight.view(
                1, 1, self.num_heads, self.vhead_dim
            )

        p_k = self.linear_pos(pos_embs).view(
            1, -1, self.num_heads, self.head_dim
        )
        # (batch, head, klen, d_k)

        q_with_bias_u = (
            query + self.pos_bias_u.view(1, 1, self.num_heads, self.head_dim)
        ).transpose(1, 2)
        # (batch, head, qlen, d_k)
        q_with_bias_v = (
            query + self.pos_bias_v.view(1, 1, self.num_heads, self.head_dim)
        ).transpose(1, 2)

        # (batch, head, qlen, klen)
        matrix_ac = torch.matmul(q_with_bias_u, key.permute(0, 2, 3, 1))
        # (batch, num_heads, klen, 2*klen-1)
        matrix_bd = torch.matmul(q_with_bias_v, p_k.permute(0, 2, 3, 1))
        matrix_bd = self.rel_shift(matrix_bd)  # shifting trick

        # if klen != qlen:
        #   import ipdb
        #  ipdb.set_trace(

        attn_score = (matrix_ac + matrix_bd) * self.scale

        # compute attention probability
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.view(1, 1, qlen, klen)
            else:
                attn_mask = attn_mask.view(-1, self.num_heads, qlen, klen)

            if attn_mask.dtype == torch.bool:
                attn_score = attn_score.masked_fill(
                    attn_mask, self.attn_fill_value
                )
            else:
                attn_score += attn_mask

        if key_padding_mask is not None:
            attn_score = attn_score.masked_fill(
                key_padding_mask.view(bsz, 1, 1, klen), self.attn_fill_value,
            )

        attn_score = F.softmax(attn_score, dim=-1)
        attn_score = self.dropout_att(attn_score)
        x = torch.matmul(
            attn_score, value.transpose(1, 2)
        )  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(bsz, -1, self.vhead_dim * self.num_heads)
        )  # (batch, time1, d_model)

        out = self.out_proj(x)
        if return_attn_weights:
            return out, attn_score
        return out


class MultiheadAttention(nn.Module):
    """ The class is a wrapper of MultiHead Attention for torch.nn.MultiHeadAttention.

    Reference: https://pytorch.org/docs/stable/nn.html

    Arguments
    ----------
    num_heads : int
        parallel attention heads.
    dropout : float
        a Dropout layer on attn_output_weights (default: 0.0).
    bias : bool
        add bias as module parameter (default: True).
    add_bias_kv : bool
        add bias to the key and value sequences at dim=0.
    add_zero_attn : bool
        add a new batch of zeros to the key and value sequences at dim=1.
    kdim : int
        total number of features in key (default: None).
    vdim : int
        total number of features in value (default: None).

    Example
    -------
    >>> inputs = torch.rand([8, 60, 512])
    >>> net = MultiheadAttention(nhead=8, d_model=inputs.shape[-1])
    >>> outputs, attn = net(inputs, inputs, inputs)
    >>> outputs.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        nhead,
        d_model,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
    ):
        super().__init__()

        self.att = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
        )

    def forward(
        self,
        query,
        key,
        value,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn_weights: Optional[torch.Tensor] = True,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        """
        Arguments
        ----------
        query : torch.Tensor
            (B, L, E) where L is the target sequence length,
            B is the batch size, E is the embedding dimension.
        key : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        value : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        key_padding_mask : torch.Tensor, optional
            (B, S) where B is the batch size, S is the source sequence
            length. If a ByteTensor is provided, the non-zero positions will
            be ignored while the position with the zero positions will be
            unchanged. If a BoolTensor is provided, the positions with the
            value of True will be ignored while the position with the value
            of False will be unchanged.
        attn_mask : torch.Tensor, optional
            2D mask (L, S) where L is the target sequence length, S is
            the source sequence length.
            3D mask (N*num_heads, L, S) where N is the batch
            size, L is the target sequence length, S is the source sequence
            length. attn_mask ensure that position i is allowed to attend the
            unmasked positions. If a ByteTensor is provided, the non-zero
            positions are not allowed to attend while the zero positions will
            be unchanged. If a BoolTensor is provided, positions with True is
            not allowed to attend while False values will be unchanged. If a
            FloatTensor is provided, it will be added to the attention weight.
        pos_embs: torch.Tensor, optional
            Positional embeddings added to the attention map of shape (L, S, E) or (L, S, 1).

        Outputs
        -------
        attn_output : torch.Tensor
            (B, L, E) where L is the target sequence length, B is the
            batch size, E is the embedding dimension.
        attn_output_weights : torch.Tensor
            (B, L, S) where B is the batch size, L is the target
            sequence length, S is the source sequence length.
        """
        # give tensors of shape (time, batch, fea)
        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        # this will be legit because of https://github.com/pytorch/pytorch/blob/5288d05cfdda85c46c4df84617fa7f37c21b10b3/torch/nn/functional.py#L4946
        # we can inject relative learnable pos embeddings directly in MHA via the attn_mask
        if pos_embs is not None:
            if attn_mask is not None:
                attn_mask += pos_embs
            else:
                attn_mask = pos_embs

        output = self.att(
            query,
            key,
            value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=return_attn_weights,
        )

        if return_attn_weights:
            output, attention_weights = output
            # reshape the output back to (batch, time, fea)
            output = output.permute(1, 0, 2)
            return output, attention_weights
        else:
            output, attention_weights = output
            output = output.permute(1, 0, 2)
            del attention_weights
            return output

class DisentangledMaskAttention(nn.Module):
    """ The class for disentangled mask attention.

    Arguments
    ----------
    num_heads : int
        parallel attention heads.
    dropout : float
        a Dropout layer on attn_output_weights (default: 0.0).
    bias : bool
        add bias as module parameter (default: True).
    add_bias_kv : bool
        add bias to the key and value sequences at dim=0.
    add_zero_attn : bool
        add a new batch of zeros to the key and value sequences at dim=1.
    kdim : int
        total number of features in key (default: None).
    vdim : int
        total number of features in value (default: None).
    clusters: int
        (default: 4)
    encoder_decoder_attention: bool
        (default: False)
    causal: bool
        (default: False)

    Example
    -------
    >>> inputs = torch.rand([8, 60, 512])
    >>> net = DisentangledMaskAttention(nhead=8, d_model=inputs.shape[-1])
    >>> outputs, attn = net(inputs, inputs, inputs)
    >>> outputs.shape
    torch.Size([8, 60, 512])
    """
    outputs_dict: Dict[str, torch.Tensor]

    def __init__(
        self,
        nhead,
        d_model,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        clusters=4,
        encoder_decoder_attention=False,
        causal=False,
        mu_grad=5
    ):
        super().__init__()

        self.nhead = nhead
        self.d_model = d_model
        self.clusters = clusters
        self.causal = causal
        self.self_attention = not encoder_decoder_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        self.eps = 1e-6

        # Attention parameters
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_output = nn.Linear(d_model, d_model)

        # Semantic mask attention parameters
        self.var = nn.Parameter(torch.full((1, 1, 1), 0.01), requires_grad=False)
        self.tok_log_prior = nn.Parameter(torch.full((nhead, clusters,), math.log(1/clusters)))
        self.tok_log_var = nn.Parameter(torch.full((nhead, clusters, d_model//nhead), math.log(1)))
        self.tok_mu = nn.Parameter(torch.cat([
            nn.init.normal_(torch.ones(nhead, 1, d_model//nhead), -0.1 + 0.1/clusters*c, 0.1) for c in range(clusters)
        ], dim=1))
        self.linear_semantic_var = nn.Linear(d_model, nhead)
        self.tok_mu.register_hook(lambda grad: mu_grad*grad)

        if self.encoder_decoder_attention:
            self.linear_semantic_var_k = nn.Linear(d_model, nhead)
        else:
            self.linear_semantic_var_k = nn.Identity()

        # Disentangled head attention parameters
        self.head_log_prior = nn.Parameter(torch.full((1, clusters,), math.log(1/clusters)))
        self.head_log_var = nn.Parameter(torch.full((1, clusters, d_model//nhead), math.log(1)))
        self.head_mu = nn.Parameter(torch.cat([
            nn.init.normal_(torch.ones(1, 1, d_model//nhead), -0.1 + 0.1/clusters*c, 0.1) for c in range(clusters)
        ], dim=1))
        self.linear_head_var = nn.Linear(d_model, nhead)
        self.head_mu.register_hook(lambda grad: mu_grad*grad)

        self.num_updates = nn.Parameter(torch.zeros((1,)), requires_grad=False)
        self.outputs_dict: Dict[str, torch.Tensor] = {}
        self.debug = False

    def init_params(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1/math.sqrt(2))

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
        
        x = x
        approx_log_var = approx_log_var
        z = self.normal_reparameterizing(x, approx_log_var.unsqueeze(-1))
        mu = mu
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
        H = self.nhead
        D_K = D//H
        L_K = key.size()[1]
        outputs = {}
        input_dtype = query.dtype

        attention_weights: Optional[torch.Tensor] = None
        if not get_attn_weights:
            attention_weights = None

        # 1. Semantic mask attention
        log_var = self.linear_semantic_var(query).view(B, L_Q, H).transpose(2, 1)

        query = query.view(B, L_Q, H, D_K).transpose(2, 1)
        if debug or self.debug:
            outputs.update({"_hidden_states": query})

        query, cluster_probs_q, cluster_loss_q, cluster_div_loss_q, _ = self.clustering(
            query,
            log_var,
            self.tok_mu,
            self.tok_log_var,
            self.tok_log_prior
        )
        query = query.transpose(2, 1).reshape(B, L_Q, D).to(input_dtype)

        if self.encoder_decoder_attention:
            log_var = self.linear_semantic_var_k(key).view(B, L_K, H).transpose(2, 1)
            key = key.view(B, L_K, H, D_K).transpose(2, 1)
            key, cluster_probs_k, cluster_loss_k, cluster_div_loss_k, _ = self.clustering(
                key,
                log_var,
                self.tok_mu,
                self.tok_log_var,
                self.tok_log_prior
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
        query = self.linear_query(query)
        key = self.linear_key(key).view(B, L_K, H, D_K).transpose(2, 1) # B x H x L x D
        value = self.linear_value(value).view(B, L_K, H, D_K).transpose(2, 1) # B x H x L x D

        # 4. Disentanglement head attention
        log_var = self.linear_head_var(query).view(B, L_Q, H).transpose(2, 1)

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

        if debug or self.debug:
            outputs.update({"_query": query})

        cluster_loss = cluster_loss + mi_cluster_loss
        cluster_div_loss = cluster_div_loss + cluster_div_loss_mi

        # 5. Softmax attention
        QK = query.matmul(key.transpose(3, 2))
        attention_weights = F.softmax(QK + padding_mask, dim=-1).float()

        if debug or self.debug:
            outputs.update({"attn": attention_weights})

        mask_attention_weights = F.normalize(attention_weights*attention_mask.float(), p=1., dim=-1, eps=1e-6)
        annealing_weight = 1 - torch.maximum(0.1*torch.ones_like(self.num_updates), torch.exp(-5e-4*self.num_updates))
        attention_weights = (1 - annealing_weight) * attention_weights + annealing_weight * mask_attention_weights

        if debug or self.debug:
            outputs.update({
                "mask": attention_mask,
                "mask_attn": attention_weights
            })

        # 6. Attention output
        attn_output = torch.einsum("bhqk,bhkd->bhqd", attention_weights.type_as(value), value)
        attn_output = attn_output.view(B, H, L_Q, D_K).transpose(2, 1).contiguous().view(B, -1, D)
        attn_output = self.linear_output(attn_output)

        del QK

        outputs.update({
            "loss_kl": cluster_loss.view(-1, 1, 1),
            "loss_mi": mi_loss.view(-1, 1, 1),
            "loss_div": cluster_div_loss.view(-1, 1, 1)
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

    def forward(
        self,
        query,
        key,
        value,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = True,
        pos_embs: Optional[torch.Tensor] = None,
    )-> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Arguments
        ----------
        query : torch.Tensor
            (B, L, E) where L is the target sequence length,
            B is the batch size, E is the embedding dimension.
        key : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        value : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        key_padding_mask : torch.Tensor, optional
            (B, S) where B is the batch size, S is the source sequence
            length. If a ByteTensor is provided, the non-zero positions will
            be ignored while the position with the zero positions will be
            unchanged. If a BoolTensor is provided, the positions with the
            value of True will be ignored while the position with the value
            of False will be unchanged.
        attn_mask : torch.Tensor, optional
            2D mask (L, S) where L is the target sequence length, S is
            the source sequence length.
            3D mask (N*num_heads, L, S) where N is the batch
            size, L is the target sequence length, S is the source sequence
            length. attn_mask ensure that position i is allowed to attend the
            unmasked positions. If a ByteTensor is provided, the non-zero
            positions are not allowed to attend while the zero positions will
            be unchanged. If a BoolTensor is provided, positions with True is
            not allowed to attend while False values will be unchanged. If a
            FloatTensor is provided, it will be added to the attention weight.
        pos_embs: torch.Tensor, optional
            Positional embeddings added to the attention map of shape (L, S, E) or (L, S, 1).

        Outputs
        -------
        attn_output : torch.Tensor
            (B, L, E) where L is the target sequence length, B is the
            batch size, E is the embedding dimension.
        attn_output_weights : torch.Tensor
            (B, L, S) where B is the batch size, L is the target
            sequence length, S is the source sequence length.
        """
        # this will be legit because of https://github.com/pytorch/pytorch/blob/5288d05cfdda85c46c4df84617fa7f37c21b10b3/torch/nn/functional.py#L4946
        # we can inject relative learnable pos embeddings directly in MHA via the attn_mask
        if pos_embs is not None:
            if attn_mask is not None:
                attn_mask += pos_embs
            else:
                attn_mask = pos_embs

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).to(query)
        else:
            key_padding_mask = torch.zeros((key.size()[0], 1, key.size()[1])).to(query)

        output, attention_weights, outputs_dict = self.disentangled_attention(
            query, key, value, key_padding_mask, return_attn_weights
        )
        self.outputs_dict.clear()
        self.outputs_dict.update(outputs_dict)

        if return_attn_weights:
            return output, attention_weights
        else:
            del attention_weights
            return output, None

class PositionalwiseFeedForward(nn.Module):
    """The class implements the positional-wise feed forward module in
    “Attention Is All You Need”.

    Arguments
    ----------
    d_ffn: int
        Hidden layer size.
    input_shape : tuple, optional
        Expected shape of the input. Alternatively use ``input_size``.
    input_size : int, optional
        Expected size of the input. Alternatively use ``input_shape``.
    dropout: float, optional
        Dropout rate.
    activation: torch.nn.Module, optional
        activation functions to be applied (Recommendation: ReLU, GELU).

    Example
    -------
    >>> inputs = torch.rand([8, 60, 512])
    >>> net = PositionalwiseFeedForward(256, input_size=inputs.shape[-1])
    >>> outputs = net(inputs)
    >>> outputs.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_ffn,
        input_shape=None,
        input_size=None,
        dropout=0.0,
        activation=nn.ReLU,
    ):
        super().__init__()

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        if input_size is None:
            input_size = input_shape[-1]

        activation = activation()

        self.ffn = nn.Sequential(
            nn.Linear(input_size, d_ffn),
            activation,
            nn.Dropout(dropout),
            nn.Linear(d_ffn//2 if isinstance(activation, torch.nn.GLU) else d_ffn, input_size),
        )

    def forward(self, x):
        # give a tensor of shap (time, batch, fea)
        x = x.permute(1, 0, 2)
        x = self.ffn(x)

        # reshape the output back to (batch, time, fea)
        x = x.permute(1, 0, 2)

        return x
