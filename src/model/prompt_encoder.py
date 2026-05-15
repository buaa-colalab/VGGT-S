"""
Modified from the SAM2 prompt encoder implementation.

Reference:
    SAM 2: Segment Anything in Images and Videos
    https://github.com/facebookresearch/sam2
"""

from typing import Type

import torch
from torch import nn

from .module import LayerNorm2d, PositionEmbeddingRandom


class PromptEncoder(nn.Module):
    """
    Prompt encoder adapted from SAM2 for ego-exo correspondence learning.

    This module encodes:
        1. Sparse prompts
            - source points
            - target points
            - source point features

        2. Dense prompts
            - source masks
            - target masks

    Outputs are used by the two-way transformer and mask decoder.

    Args:
        embed_dim (int):
            Embedding feature dimension.

        img_size (int):
            Input image resolution.

        mask_in_chans (int):
            Intermediate channels for mask encoding.

        sa_img_attn_ratio (int):
            Self-attention channel expansion ratio.

        sa_tokens (int):
            Spatial token resolution for self-attention.

        image_embedding_size (int):
            Feature map resolution.

        activation (Type[nn.Module]):
            Activation layer.
            Default: nn.GELU.
    """

    def __init__(
        self,
        embed_dim: int,
        img_size: int,
        mask_in_chans: int,
        sa_img_attn_ratio: int = 7,
        sa_tokens: int = 37,
        image_embedding_size: int = 259,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Initialize prompt encoder.

        Components:
            - Random Fourier positional encoding
            - Source-target image embeddings
            - Point-type embeddings
            - Mask downscaling encoder
        """

        super().__init__()

        self.embed_dim = embed_dim
        self.img_size = img_size
        self.sa_tokens = sa_tokens
        self.image_embedding_size = image_embedding_size

        # Mask encoder input size
        self.mask_input_size = image_embedding_size * 2

        # ======================== positional encoding ========================

        # Cross-attention positional encoding
        self.ca_pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # Self-attention positional encoding
        self.sa_pe_layer = PositionEmbeddingRandom(embed_dim *
                                                   sa_img_attn_ratio // 2)

        # ================== image identity embeddings =================

        # Cross-attention image identifiers
        self.ca_src_img_pe = nn.Embedding(1, embed_dim)
        self.ca_tar_img_pe = nn.Embedding(1, embed_dim)

        # Self-attention image identifiers
        self.sa_src_img_pe = nn.Embedding(1, embed_dim * sa_img_attn_ratio)

        self.sa_tar_img_pe = nn.Embedding(1, embed_dim * sa_img_attn_ratio)

        # ================ point identity embeddings ===================

        self.src_point_pe = nn.Embedding(1, embed_dim)

        self.tar_point_pe = nn.Embedding(1, embed_dim)

        # ======================== mask encoder ========================

        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )

    def get_dense_ca_pe(self) -> torch.Tensor:
        """
        Generate dense positional encoding for cross-attention.

        Returns:
            torch.Tensor:
                Cross-attention positional encoding.

                Shape:
                    [1, C, H, W * 2]
        """

        # Base positional encoding
        base_pe = self.ca_pe_layer(
            (self.img_size // 2, self.img_size // 2)).unsqueeze(0)

        # Add source-target identity embeddings
        ca_pe = torch.cat([
            base_pe + self.ca_src_img_pe.weight.view(1, -1, 1, 1),
            base_pe + self.ca_tar_img_pe.weight.view(1, -1, 1, 1),
        ],
                          dim=-1)

        return ca_pe

    # Output:
    # [1, 128, 259, 259 * 2]

    def get_dense_sa_pe(self) -> torch.Tensor:
        """
        Generate dense positional encoding for self-attention.

        Returns:
            torch.Tensor:
                Self-attention positional encoding.

                Shape:
                    [1, C_sa, H_sa, W_sa * 2]
        """

        # Base positional encoding
        base_pe = self.sa_pe_layer(
            (self.sa_tokens, self.sa_tokens)).unsqueeze(0)

        # Add source-target identity embeddings
        sa_pe = torch.cat([
            base_pe + self.sa_src_img_pe.weight.view(1, -1, 1, 1),
            base_pe + self.sa_tar_img_pe.weight.view(1, -1, 1, 1),
        ],
                          dim=-1)

        return sa_pe

    # Output:
    # [1, 896, 37, 37 * 2]

    def forward(
            self,
            src_points_b: torch.Tensor,  # [B, N, 2]
            tar_points_b: torch.Tensor,  # [B, N, 2]
            src_mask_b: torch.Tensor,  # [B, 1, H, W]
            src_points_feat_b,
            tar_mask_b=None,  # [B, 1, H, W]
    ):
        """
        Encode sparse and dense prompts.

        Sparse prompts:
            - source points
            - target points
            - source point features

        Dense prompts:
            - source masks
            - target masks

        Args:
            src_points_b (torch.Tensor):
                Source prompt points.
                Shape: [B, N, 2]

            tar_points_b (torch.Tensor):
                Target projected points.
                Shape: [B, N, 2]

            src_mask_b (torch.Tensor):
                Source binary masks.
                Shape: [B, 1, H, W]

            src_points_feat_b (torch.Tensor):
                VGGT point feature embeddings.

            tar_mask_b (torch.Tensor, optional):
                Target refinement masks.
                Shape: [B, 1, H, W]

        Returns:
            sparse_embeddings_b (torch.Tensor):
                Sparse prompt embeddings.
                Shape: [B, N * 3, C]

            src_dense_embeddings_img_b (torch.Tensor):
                Source dense mask embeddings.
                Shape: [B, C, H', W']

            tar_dense_embeddings_img_b (torch.Tensor):
                Target dense mask embeddings.
                Shape: [B, C, H', W']
        """

        # ======================== point encoding ========================

        # Shift coordinates to pixel center
        src_points_b = src_points_b + 0.5
        tar_points_b = tar_points_b + 0.5

        # Encode source points
        src_points_embed = (self.ca_pe_layer.forward_with_coords(
            src_points_b, (self.img_size, self.img_size)))  # [B, N, C]

        # Encode target points
        tar_points_embed = (self.ca_pe_layer.forward_with_coords(
            tar_points_b, (self.img_size, self.img_size)))  # [B, N, C]

        # Add point-type embeddings
        src_points_embed += (self.src_point_pe.weight.view(1, 1, -1))

        tar_points_embed += (self.tar_point_pe.weight.view(1, 1, -1))

        # Add source identity to VGGT features
        src_points_feat_b += (self.src_point_pe.weight.view(1, 1, -1))

        # Concatenate sparse embeddings
        sparse_embeddings_b = torch.cat(
            [src_points_embed, tar_points_embed, src_points_feat_b],
            dim=1)  # [B, N * 3, C]

        # ======================== mask encoding ========================

        # Encode source mask
        src_dense_embeddings_img_b = (self.mask_downscaling(src_mask_b)
                                      )  # [B, 128, 259, 259]

        # Encode target refinement mask
        if tar_mask_b is not None:

            tar_dense_embeddings_img_b = (self.mask_downscaling(tar_mask_b)
                                          )  # [B, 128, 259, 259]

        else:
            tar_dense_embeddings_img_b = None

        return (sparse_embeddings_b, src_dense_embeddings_img_b,
                tar_dense_embeddings_img_b)
