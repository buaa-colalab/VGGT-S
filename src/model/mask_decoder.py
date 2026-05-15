"""
Modified from the SAM2 mask decoder implementation.

Reference:
    SAM 2: Segment Anything in Images and Videos
    https://github.com/facebookresearch/sam2
"""

from typing import List, Tuple, Type

import torch
from torch import nn

from .module import MLP, LayerNorm2d


class MaskDecoder(nn.Module):
    """
    Mask decoder adapted from SAM2 for ego-exo correspondence segmentation.

    Pipeline:
        1. Concatenate mask tokens and sparse point embeddings.
        2. Fuse source-target image features.
        3. Run transformer-based cross-attention reasoning.
        4. Upsample target image features.
        5. Predict masks using hypernetwork-generated mask weights.

    Args:
        transformer_dim (int):
            Transformer feature dimension.

        transformer (nn.Module):
            Cross-attention transformer module.

        activation (Type[nn.Module]):
            Activation layer used in upscaling blocks.
            Default: nn.GELU.
    """

    def __init__(
        self,
        transformer_dim: int,
        transformer: nn.Module,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Initialize the mask decoder.

        Components:
            - Learnable mask tokens
            - Feature upscaling decoder
            - Hypernetwork MLPs for mask prediction
            - Cross-attention transformer
        """

        super().__init__()

        self.transformer_dim = transformer_dim
        self.transformer = transformer

        # Number of output mask tokens
        self.num_mask_tokens = 1

        # Learnable mask tokens
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # Feature upscaling decoder
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim,
                               transformer_dim // 4,
                               kernel_size=2,
                               stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4,
                               transformer_dim // 8,
                               kernel_size=2,
                               stride=2),
            activation(),
        )

        # Hypernetwork MLPs used for mask generation
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
            for i in range(self.num_mask_tokens)
        ])

    def forward(
        self,
        feat_map_b: torch.Tensor,  # [B, 2, 128, 259, 259]
        ca_img_pe: torch.Tensor,  # [1, 128, 259, 259 * 2]
        sa_img_pe: torch.Tensor,  # [1, 896, 37, 37 * 2]
        sparse_embeddings_b: torch.Tensor,  # [B, N * 2, 128]
        src_dense_embeddings_img_b: torch.Tensor,  # [B, 128, 259, 259]
        tar_dense_embeddings_img_b: torch.Tensor,  # [B, 128, 259, 259]
        mask_token_b=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict segmentation masks from source-target image features.

        Args:
            feat_map_b (torch.Tensor):
                Source-target image feature maps.
                Shape: [B, 2, C, H, W].

            ca_img_pe (torch.Tensor):
                Cross-attention positional encoding.
                Shape: [1, C, H, W * 2].

            sa_img_pe (torch.Tensor):
                Self-attention positional encoding.
                Shape: [1, C_sa, H_sa, W_sa * 2].

            sparse_embeddings_b (torch.Tensor):
                Sparse point embeddings.
                Shape: [B, N * 2, C].

            src_dense_embeddings_img_b (torch.Tensor):
                Source dense image embeddings.
                Shape: [B, C, H, W].

            tar_dense_embeddings_img_b (torch.Tensor):
                Target dense image embeddings.
                Shape: [B, C, H, W].

            mask_token_b (torch.Tensor, optional):
                External mask tokens.
                Shape: [B, M, C].

        Returns:
            masks (torch.Tensor):
                Predicted segmentation masks.
                Shape: [B, M, H, W].
        """

        # ======================== token preparation ========================

        if mask_token_b is None:

            # Default learnable mask tokens
            output_tokens = self.mask_tokens.weight  # [M, C]

            output_tokens = output_tokens.unsqueeze(0).expand(
                sparse_embeddings_b.size(0), -1, -1)  # [B, M, C]

        else:
            output_tokens = mask_token_b

        # Concatenate:
        #   - mask tokens
        #   - sparse point embeddings
        tokens = torch.cat((output_tokens, sparse_embeddings_b),
                           dim=1)  # [B, M + N * 2, C]

        # ================= image feature fusion ====================

        src_img_emb = feat_map_b[:, 0, ...]  # [B, C, H, W]
        tar_img_emb = feat_map_b[:, 1, ...]  # [B, C, H, W]

        # Concatenate source-target features along width dimension
        image_embeddings = torch.cat([src_img_emb, tar_img_emb],
                                     dim=-1)  # [B, C, H, W * 2]

        src = image_embeddings

        b, c, h, w2 = src.shape

        # =================== transformer reasoning ===================

        hs, src = self.transformer(src, ca_img_pe, sa_img_pe, tokens,
                                   src_dense_embeddings_img_b,
                                   tar_dense_embeddings_img_b)

        # Extract mask token outputs
        mask_tokens_out = hs[:, 0:self.num_mask_tokens, :]

        # ================= feature reconstruction ==================

        src = src.transpose(1, 2).view(b, c, h, w2)

        updated_img_feat_b = src

        # Recover source / target feature maps
        updated_src_b = updated_img_feat_b[:, :, :, :w2 // 2]

        updated_tar_b = updated_img_feat_b[:, :, :, w2 // 2:]

        updated_img_feat_b = torch.stack([updated_src_b, updated_tar_b],
                                         dim=1)  # [B, 2, C, H, W]

        # Use target-side features for mask prediction
        src = src[:, :, :, w2 // 2:]

        # ======================== feature upscaling ========================

        upscaled_embedding = self.output_upscaling(src)

        # ================= hypernetwork mask generation ===================

        hyper_in_list: List[torch.Tensor] = []

        for i in range(self.num_mask_tokens):

            hyper_in_list.append(self.output_hypernetworks_mlps[i](
                mask_tokens_out[:, i, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)

        b, c, h, w = upscaled_embedding.shape

        # Dynamic mask prediction
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(
            b, -1, h, w)  # [B, M, H, W]

        return masks
