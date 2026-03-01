"""
graph_encoder.py — Graph Attention Encoder for the Attention Model

This file implements the ENCODER described in Section I of the LaTeX document.
It takes raw node features and produces context-aware d-dimensional embeddings
for every node in the graph, using a Transformer-style architecture.

HIGH-LEVEL FLOW (corresponds to Section I of the LaTeX):
    1. Initial linear embedding   → Eq. (1)-(2):  h_i^(0) = W·[features] + b
    2. L=3 Multi-Head Attention layers → Eq. (3)-(8): self-attention + FF + skip + BN
    3. Graph embedding (mean pool) → h̄ = mean(h_i^(L))

Classes:
    - SkipConnection:           Wraps a module with a residual/skip connection
    - MultiHeadAttention:       Core MHA computation — Eq. (5)-(8)
    - Normalization:            Batch or Instance normalization
    - MultiHeadAttentionLayer:  One full encoder layer — Eq. (3)-(4)
    - GraphAttentionEncoder:    Top-level encoder stacking L layers
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import math


class SkipConnection(nn.Module):
    """
    Implements the residual (skip) connection used in Eq. (3) and (4):
        output = input + module(input)

    In the encoder, this wraps both the MHA sublayer and the FF sublayer,
    so that each sublayer's output is added back to its input before normalization.
    """

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        # Residual connection: input + sublayer(input)
        return input + self.module(input)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (MHA) — implements Eq. (5)-(8) from the LaTeX.

    For M=8 heads, each with key dimension d_k = d/M = 16:

        Eq. (5): Q_i^m = W_Q^m · h_i,   K_j^m = W_K^m · h_j,   V_j^m = W_V^m · h_j
        Eq. (6): α_ij^m = softmax( Q_i^m · K_j^m / √d_k )
        Eq. (7): head_i^m = Σ_j α_ij^m · V_j^m
        After Eq. (7): MHA(h_i) = W_O · [head_i^1 ; ... ; head_i^M]   (concatenate + project)

    Parameters:
        n_heads:    M = 8 attention heads
        input_dim:  d = 128 (embedding dimension of input)
        embed_dim:  d = 128 (output dimension, must equal input_dim for residual)
        val_dim:    d_k = d/M = 16 (value dimension per head)
        key_dim:    d_k = 16 (key dimension per head, same as val_dim by default)
    """
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads  # d_k = d / M = 128 / 8 = 16
        if key_dim is None:
            key_dim = val_dim  # key_dim = val_dim = 16

        self.n_heads = n_heads        # M = 8
        self.input_dim = input_dim    # d = 128
        self.embed_dim = embed_dim    # d = 128 (output dim)
        self.val_dim = val_dim        # d_k = 16
        self.key_dim = key_dim        # d_k = 16

        # Eq. (6): scaling factor 1/√d_k from "Attention Is All You Need"
        self.norm_factor = 1 / math.sqrt(key_dim)

        # Eq. (5): Learnable projection matrices for each head
        # W_Q^m, W_K^m, W_V^m  — shape: (M, d, d_k) i.e. (8, 128, 16)
        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))  # W_Q^m
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))    # W_K^m
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))    # W_V^m

        # After Eq. (7): W_O projects concatenated heads back to d=128
        # Shape: (M, d_k, d) i.e. (8, 16, 128), reshaped to (128, 128) during forward
        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))  # W_O

        self.init_parameters()

    def init_parameters(self):
        """Initialize all parameters with uniform distribution scaled by 1/√(fan_in)."""
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
        Compute multi-head attention.

        In the ENCODER, this is called as self-attention: q = h (all node embeddings).
        In the DECODER glimpse, q is the single query vector and h is the node embeddings.

        Args:
            q:    queries — (batch_size, n_query, d)       e.g. (512, 21, 128) in encoder
            h:    keys/values — (batch_size, graph_size, d) (defaults to q for self-attention)
            mask: (batch_size, n_query, graph_size) — True where attention is blocked

        Returns:
            out:  (batch_size, n_query, embed_dim) — attended output
        """
        if h is None:
            h = q  # Self-attention: queries = keys = values = node embeddings

        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        # Flatten for efficient batch matrix multiplication
        # (B*N, d) — treats each node embedding independently for the linear projection
        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        shp = (self.n_heads, batch_size, graph_size, -1)    # reshape target for K, V
        shp_q = (self.n_heads, batch_size, n_query, -1)     # reshape target for Q

        # ====================================================================
        # Eq. (5): Compute Q, K, V for all heads simultaneously
        # ====================================================================
        # qflat (B*n_query, d) × W_query (M, d, d_k) → (M, B*n_query, d_k) → (M, B, n_query, d_k)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)   # (8, B, n_query, 16)
        K = torch.matmul(hflat, self.W_key).view(shp)        # (8, B, N+1, 16)
        V = torch.matmul(hflat, self.W_val).view(shp)        # (8, B, N+1, 16)

        # ====================================================================
        # Eq. (6): Compute attention weights  α_ij^m = softmax(Q·K^T / √d_k)
        # ====================================================================
        # Q (8, B, n_query, 16) × K^T (8, B, 16, N+1) → compatibility (8, B, n_query, N+1)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Apply mask: set infeasible positions to -inf so softmax gives 0 weight
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = F.softmax(compatibility, dim=-1)  # α_ij^m — (8, B, n_query, N+1)

        # Fix NaN from softmax when all positions are masked (node has no valid neighbors)
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        # ====================================================================
        # Eq. (7): Compute attention output  head_i^m = Σ_j α_ij^m · V_j^m
        # ====================================================================
        # attn (8, B, n_query, N+1) × V (8, B, N+1, 16) → heads (8, B, n_query, 16)
        heads = torch.matmul(attn, V)

        # ====================================================================
        # After Eq. (7): Concatenate heads and project  MHA = W_O · [head^1; ...; head^M]
        # ====================================================================
        # Rearrange: (8, B, n_query, 16) → (B, n_query, 8, 16) → (B*n_query, 128)
        # Then multiply by W_out reshaped to (128, 128) → (B, n_query, 128)
        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out  # (B, n_query, d) — the MHA output for each node


class Normalization(nn.Module):
    """
    Batch Normalization layer used in Eq. (3) and (4):
        ĥ_i = BN( h_i + MHA(...) )
        h_i  = BN( ĥ_i + FF(ĥ_i) )

    Applied per-feature across the batch dimension. The input shape
    (B, N+1, d) is reshaped to (B*(N+1), d) for BatchNorm1d.
    """

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,       # Default: batch normalization
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

    def init_parameters(self):
        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            # Reshape (B, N+1, d) → (B*(N+1), d), apply BN, reshape back
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttentionLayer(nn.Sequential):
    """
    One complete encoder layer — implements Eq. (3) and (4) together:

        Eq. (3): ĥ_i^(ℓ) = BN( h_i^(ℓ-1) + MHA({h_j^(ℓ-1)}) )     [attention + skip + BN]
        Eq. (4): h_i^(ℓ)  = BN( ĥ_i^(ℓ)  + FF(ĥ_i^(ℓ)) )          [feed-forward + skip + BN]

    The layer is composed as a Sequential of 4 modules:
        1. SkipConnection(MHA)     — residual around multi-head self-attention
        2. Normalization(BN)       — batch norm after attention sublayer
        3. SkipConnection(FF)      — residual around feed-forward network
        4. Normalization(BN)       — batch norm after feed-forward sublayer

    The FF network is: Linear(d→512) → ReLU → Linear(512→d)
    """

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            # Sublayer 1: Skip + MHA  →  h + MHA(h)  [first half of Eq. (3)]
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            # BN after attention sublayer  [second half of Eq. (3)]
            Normalization(embed_dim, normalization),
            # Sublayer 2: Skip + FF  →  ĥ + FF(ĥ)  [first half of Eq. (4)]
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),  # d=128 → 512
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)   # 512 → d=128
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            # BN after feed-forward sublayer  [second half of Eq. (4)]
            Normalization(embed_dim, normalization)
        )


class GraphAttentionEncoder(nn.Module):
    """
    Top-level encoder — implements the full pipeline of Section I:

        1. Initial embedding:  Eq. (1)-(2)
           - This class does NOT create the init_embed layers itself; they are
             created in AttentionModel._init_embed() which handles depot vs. sensor
             embeddings separately. This class receives already-embedded input when
             node_dim is None, OR applies a generic Linear(node_dim → d) if node_dim
             is provided.

        2. L=3 attention layers:  Eq. (3)-(4) applied sequentially
           - Each layer is a MultiHeadAttentionLayer (see above)

        3. Graph embedding:  h̄ = (1/(N+1)) Σ h_i^(L)
           - Mean pooling over all node embeddings

    Args:
        n_heads:              M = 8 attention heads
        embed_dim:            d = 128 embedding dimension
        n_layers:             L = 3 encoder layers
        node_dim:             Input feature dimension (None if pre-embedded)
        normalization:        'batch' (default) or 'instance'
        feed_forward_hidden:  Hidden dim of FF sublayer (512)

    Returns (in forward):
        h:        (B, N+1, d)  — per-node embeddings after L layers
        h.mean:   (B, d)       — graph-level embedding h̄
    """
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphAttentionEncoder, self).__init__()

        # Optional initial linear projection: maps raw features → d-dim embedding
        # For BC-CSP, this is None because _init_embed in AttentionModel handles it
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        # Stack L=3 MultiHeadAttentionLayers sequentially
        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, N+1, node_dim) raw features, or (B, N+1, d) pre-embedded features
            mask: Not yet supported

        Returns:
            h:          (B, N+1, d)  — final node embeddings h_i^(L)
            h.mean(1):  (B, d)       — graph embedding h̄ = mean over all nodes
        """
        assert mask is None, "TODO mask not yet supported!"

        # Initial embedding (if init_embed exists; for BC-CSP it's pre-applied)
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x

        # Pass through L=3 attention layers: Eq. (3)-(4) applied L times
        h = self.layers(h)

        return (
            h,             # (B, N+1, d) — per-node embeddings after L layers
            h.mean(dim=1), # (B, d) — graph embedding h̄ = (1/(N+1)) Σ h_i^(L)
        )