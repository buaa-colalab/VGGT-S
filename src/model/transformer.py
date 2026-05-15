"""
Modified from the SAM2 transformer implementation.

Reference:
    SAM 2: Segment Anything in Images and Videos
    https://github.com/facebookresearch/sam2
"""

from typing import Tuple, Type

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .module import MLP


class TwoWayTransformer(nn.Module):
    """
    Two-way transformer adapted from SAM2 for ego-exo mask propagation
    and cross-view correspondence learning.

    Architecture:
        Repeated TwoWayAttentionBlock layers followed by a final
        token-to-image attention layer.

    Args:
        depth (int):
            Number of transformer blocks.

        embedding_dim (int):
            Feature embedding dimension.

        num_heads (int):
            Number of attention heads.

        sa_img_attn_ratio (int):
            Spatial downsampling ratio used for image self-attention.

        activation (Type[nn.Module]):
            Activation function.
            Default: nn.ReLU.
    """

    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        sa_img_attn_ratio: int = 7,
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:

        super().__init__()

        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Transformer blocks
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(embedding_dim=embedding_dim,
                                     num_heads=num_heads,
                                     activation=activation,
                                     skip_first_layer_pe=(i == 0),
                                     sa_img_attn_ratio=sa_img_attn_ratio))

        # Final token-to-image attention
        self.final_attn_token_to_image = Attention(embedding_dim, num_heads)

        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
            self,
            image_embedding: Tensor,  # [B, 128, 259, 259 * 2]
            ca_image_pe: Tensor,  # [1, 128, 259, 259 * 2]
            sa_image_pe: Tensor,  # [1, 896, 37, 37 * 2]
            point_embedding: Tensor,  # [B, N, 128]
            src_mask_emb: Tensor,  # [B, 128, 259, 259]
            tar_mask_emb: Tensor,  # [B, 128, 259, 259]
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the two-way transformer.

        Args:
            image_embedding (Tensor):
                Source-target image embeddings.

            ca_image_pe (Tensor):
                Cross-attention positional encoding.

            sa_image_pe (Tensor):
                Self-attention positional encoding.

            point_embedding (Tensor):
                Sparse prompt embeddings.

            src_mask_emb (Tensor):
                Source mask embeddings.

            tar_mask_emb (Tensor):
                Target refinement mask embeddings.

        Returns:
            queries (Tensor):
                Updated sparse token embeddings.

            keys (Tensor):
                Updated image embeddings.
        """

        # ================== flatten image features ==================

        # BxCxHxW -> BxHWxC
        bs, c, h, w2 = image_embedding.shape

        image_embedding = (image_embedding.flatten(2).permute(0, 2, 1))

        ca_image_pe = (ca_image_pe.flatten(2).permute(0, 2, 1))

        sa_image_pe = (sa_image_pe.flatten(2).permute(0, 2, 1))

        # Sparse prompt tokens
        queries = point_embedding

        # Dense image tokens
        keys = image_embedding

        # ======================== transformer blocks ========================

        for layer in self.layers:

            queries, keys = layer(queries=queries,
                                  keys=keys,
                                  query_pe=point_embedding,
                                  ca_key_pe=ca_image_pe,
                                  sa_key_pe=sa_image_pe,
                                  src_mask_emb=src_mask_emb,
                                  tar_mask_emb=tar_mask_emb)

        # =============== final token-to-image attention =================

        q = queries + point_embedding
        k = keys + ca_image_pe

        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)

        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    """
    Core transformer block for bidirectional interaction between:

        - Sparse prompt tokens
        - Dense image embeddings

    Adapted from SAM2 with additional:
        - Image self-attention branch
        - Dual-view interaction
        - Source-target mask conditioning
        - Compressed image attention

    Pipeline:
        1. Image self-attention
        2. Token self-attention
        3. Token-to-image cross attention
        4. Image-to-token cross attention
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        sa_img_attn_ratio: int = 7,
        activation: Type[nn.Module] = nn.ReLU,
        skip_first_layer_pe: bool = False,
    ) -> None:

        super().__init__()

        # ================== image self-attention =================

        # Spatial downsampling
        self.downsampling = nn.Conv2d(in_channels=128,
                                      out_channels=sa_img_attn_ratio * 128,
                                      kernel_size=sa_img_attn_ratio,
                                      stride=sa_img_attn_ratio,
                                      padding=0)

        # Spatial upsampling
        self.upsampling = nn.ConvTranspose2d(in_channels=sa_img_attn_ratio *
                                             128,
                                             out_channels=128,
                                             kernel_size=sa_img_attn_ratio,
                                             stride=sa_img_attn_ratio,
                                             padding=0,
                                             output_padding=0)

        # Self-attention over compressed image tokens
        self.self_attn_img = Attention(
            sa_img_attn_ratio * 128,
            num_heads,
            downsample_rate=sa_img_attn_ratio / 2,
        )

        self.norm0 = nn.LayerNorm(sa_img_attn_ratio * 128)

        self.norm_nf = nn.LayerNorm(embedding_dim)

        # Image FFN
        self.img_mlp = MLP(sa_img_attn_ratio * 128,
                           sa_img_attn_ratio * 128 * 4,
                           sa_img_attn_ratio * 128,
                           num_layers=2,
                           activation=activation)

        self.norm_ffn = nn.LayerNorm(sa_img_attn_ratio * 128)

        # ================== token self-attention =================

        self.self_attn_token = Attention(embedding_dim, num_heads)

        self.norm1 = nn.LayerNorm(embedding_dim)

        # ================== token-to-image cross attention ===============

        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads)

        self.norm2 = nn.LayerNorm(embedding_dim)

        # Token FFN
        self.token_mlp = MLP(embedding_dim,
                             embedding_dim * 4,
                             embedding_dim,
                             num_layers=2,
                             activation=activation)

        self.norm3 = nn.LayerNorm(embedding_dim)

        # =============== image-to-token cross attention =================

        self.norm4 = nn.LayerNorm(embedding_dim)

        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads)

        # Whether to inject mask prior in first layer
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries: Tensor, keys: Tensor, query_pe: Tensor,
                ca_key_pe: Tensor, sa_key_pe: Tensor, src_mask_emb: Tensor,
                tar_mask_emb: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of a two-way attention block.

        Args:
            queries (Tensor):
                Sparse token embeddings.

            keys (Tensor):
                Dense image embeddings.

            query_pe (Tensor):
                Sparse positional embeddings.

            ca_key_pe (Tensor):
                Cross-attention positional encoding.

            sa_key_pe (Tensor):
                Self-attention positional encoding.

            src_mask_emb (Tensor):
                Source mask embeddings.

            tar_mask_emb (Tensor):
                Target mask embeddings.

        Returns:
            queries (Tensor):
                Updated sparse tokens.

            keys (Tensor):
                Updated image tokens.
        """

        # ============================================================
        # 1. Image self-attention
        # ============================================================

        B, C, H, W = src_mask_emb.shape

        ori_keys = keys  # [B, L, C]

        # Recover spatial layout
        keys = (keys.permute(0, 2, 1).reshape(B, C, H, W * 2))

        # Split source-target embeddings
        src_img_emb = keys[:, :, :, :W]
        tar_img_emb = keys[:, :, :, W:]

        # Inject mask prior only in first layer
        if self.skip_first_layer_pe:

            src_img_emb = src_img_emb + src_mask_emb

            if tar_mask_emb is not None:
                tar_img_emb = tar_img_emb + tar_mask_emb

            ori_keys = torch.cat([src_img_emb, tar_img_emb], dim=-1)

            ori_keys = (ori_keys.flatten(2).permute(0, 2, 1))

        # ======================== downsample ========================

        src_img_emb = self.downsampling(src_img_emb)
        tar_img_emb = self.downsampling(tar_img_emb)

        imgs_emb = torch.cat([src_img_emb, tar_img_emb], dim=-1)

        B, C, H, W2 = imgs_emb.shape

        # ======================== self-attention ========================

        keys = imgs_emb.flatten(2).permute(0, 2, 1)

        k = keys + sa_key_pe

        attn_out = self.self_attn_img(q=k, k=k, v=keys)

        keys = keys + attn_out
        keys = self.norm0(keys)

        # ======================== image FFN ========================

        mlp_out = self.img_mlp(keys)

        keys = keys + mlp_out
        keys = self.norm_ffn(keys)

        # ======================== upsample ========================

        keys = (keys.permute(0, 2, 1).reshape(B, C, H, W2))

        src_img_emb = keys[:, :, :, :W2 // 2]
        tar_img_emb = keys[:, :, :, W2 // 2:]

        src_img_emb = self.upsampling(src_img_emb)
        tar_img_emb = self.upsampling(tar_img_emb)

        # Merge source-target
        imgs_emb = torch.cat([src_img_emb, tar_img_emb], dim=-1)

        keys = imgs_emb.flatten(2).permute(0, 2, 1)

        # Residual connection
        keys = keys + ori_keys
        keys = self.norm_nf(keys)

        # ============================================================
        # 2. Token self-attention
        # ============================================================

        if self.skip_first_layer_pe:

            queries = self.self_attn_token(q=queries, k=queries, v=queries)

        else:

            q = queries + query_pe

            attn_out = self.self_attn_token(q=q, k=q, v=queries)

            queries = queries + attn_out

        queries = self.norm1(queries)

        # ============================================================
        # 3. Token-to-image cross attention
        # ============================================================

        q = queries + query_pe
        k = keys + ca_key_pe

        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)

        queries = queries + attn_out
        queries = self.norm2(queries)

        # Token FFN
        mlp_out = self.token_mlp(queries)

        queries = queries + mlp_out
        queries = self.norm3(queries)

        # ============================================================
        # 4. Image-to-token cross attention
        # ============================================================

        q = queries + query_pe
        k = keys + ca_key_pe

        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)

        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    Multi-head attention layer with optional channel downsampling.

    Adapted from SAM2.

    Supports:
        - self-attention
        - cross-attention
        - compressed attention computation

    Args:
        embedding_dim (int):
            Input embedding dimension.

        num_heads (int):
            Number of attention heads.

        downsample_rate (float):
            Channel reduction ratio.

        dropout (float):
            Attention dropout probability.

        kv_in_dim (int, optional):
            Key-value input dimension.
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        downsample_rate=1,
        dropout: float = 0.0,
        kv_in_dim: int = None,
    ) -> None:

        super().__init__()

        self.embedding_dim = embedding_dim

        self.kv_in_dim = (kv_in_dim
                          if kv_in_dim is not None else embedding_dim)

        # Internal projection dimension
        self.internal_dim = embedding_dim // downsample_rate
        self.internal_dim = int(self.internal_dim)

        self.num_heads = num_heads

        assert (self.internal_dim %
                num_heads == 0), "num_heads must divide embedding_dim."

        # Linear projections
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)

        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)

        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)

        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.dropout_p = dropout

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        """
        Split channels into multi-head representation.

        Input:
            [B, N, C]

        Output:
            [B, Heads, N, C_per_head]
        """

        b, n, c = x.shape

        x = x.reshape(b, n, num_heads, c // num_heads)

        return x.transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        """
        Merge multi-head representation back.

        Input:
            [B, Heads, N, C_per_head]

        Output:
            [B, N, C]
        """

        b, n_heads, n_tokens, c_per_head = x.shape

        x = x.transpose(1, 2)

        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        Multi-head scaled dot-product attention.

        Args:
            q (Tensor):
                Query tensor.

            k (Tensor):
                Key tensor.

            v (Tensor):
                Value tensor.

        Returns:
            Tensor:
                Attention output.
        """

        # ======================== projections ========================

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # ======================== split heads ========================

        q = self._separate_heads(q, self.num_heads)

        k = self._separate_heads(k, self.num_heads)

        v = self._separate_heads(v, self.num_heads)

        # ======================== attention ========================

        dropout_p = (self.dropout_p if self.training else 0.0)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        # ======================== merge heads ========================

        out = self._recombine_heads(out)

        out = self.out_proj(out)

        return out
