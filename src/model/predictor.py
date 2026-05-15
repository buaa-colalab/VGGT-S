"""
Modified from the SAM2 predictor implementation.

Reference:
    SAM 2: Segment Anything in Images and Videos
    https://github.com/facebookresearch/sam2
"""

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from utils import recover_square_crop_tensor

from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .transformer import TwoWayTransformer


class Predictor(nn.Module):
    """
    Ego-exo correspondence segmentation predictor.

    This module combines:
        1. Prompt encoding
        2. Cross-view transformer reasoning
        3. SAM2-style mask decoding

    Pipeline:
        1. Encode source/target prompts.
        2. Fuse sparse and dense embeddings.
        3. Decode target mask from cross-view features.
        4. Upsample low-resolution masks.
        5. Recover masks to original image coordinates.

    Args:
        img_size (int):
            Input image resolution.

        sa_tokens (int):
            Number of self-attention tokens.

        resume_ckpt (str, optional):
            Checkpoint path for model initialization.

        embed_dim (int):
            Feature embedding dimension.

        blocks (int):
            Number of transformer blocks.
    """

    def __init__(self,
                 img_size=518,
                 sa_tokens=37,
                 resume_ckpt=None,
                 embed_dim=128,
                 blocks=2):
        """
        Initialize predictor network.

        Components:
            - Prompt encoder
            - Two-way transformer
            - Mask decoder
        """

        super().__init__()

        # Self-attention downsampling ratio
        sa_img_attn_ratio = (img_size // 2 // sa_tokens)

        # ======================== prompt encoder ========================

        self.prompt_encoder = PromptEncoder(
            embed_dim=embed_dim,
            img_size=img_size,
            mask_in_chans=16,
            sa_img_attn_ratio=sa_img_attn_ratio,
            image_embedding_size=img_size // 2,
            sa_tokens=sa_tokens)

        # ======================== mask decoder ========================

        self.mask_decoder = MaskDecoder(
            transformer_dim=embed_dim,
            transformer=TwoWayTransformer(depth=blocks,
                                          embedding_dim=embed_dim,
                                          num_heads=8,
                                          sa_img_attn_ratio=sa_img_attn_ratio))

        # ======================== checkpoint loading ========================

        if resume_ckpt is not None:
            self._load_resume(resume_ckpt)

    def forward(
        self,
        feat_map_b,  # [B, 2, 128, 518, 518]
        src_points_b,  # [B, N, 2]
        tar_points_b,  # [B, N, 2]
        src_mask_b,  # [B, H, W]
        tar_mask_b,  # [B, H, W]
        src_points_feat_b,
        ret_logits=True,
        meta=None,
        tar_square=None,
    ):
        """
        Predict target masks from source-target correspondence features.

        Args:
            feat_map_b (torch.Tensor):
                Source-target feature maps.
                Shape: [B, 2, C, H, W]

            src_points_b (torch.Tensor):
                Source prompt points.
                Shape: [B, N, 2]

            tar_points_b (torch.Tensor):
                Target projected points.
                Shape: [B, N, 2]

            src_mask_b (torch.Tensor):
                Source binary masks.
                Shape: [B, H, W]

            tar_mask_b (torch.Tensor):
                Target mask input for iterative refinement.
                Shape: [B, H, W]

            src_points_feat_b (torch.Tensor):
                Source point feature embeddings.

            ret_logits (bool):
                Whether to return logits directly.

            meta:
                Crop recovery metadata.

            tar_square (int):
                Recovery target resolution.

        Returns:
            If ret_logits=True:
                pred_mask_b (torch.Tensor):
                    Predicted mask logits.
                    Shape: [B, 1, H, W]

                low_res_mask_b (torch.Tensor):
                    Low-resolution mask logits.

            Else:
                pred_mask_b:
                    Binary masks.

                low_res_mask_b:
                    Low-resolution logits.

                pred_mask_recover_b:
                    Recovered original-resolution masks.

                pred_points_recover_b:
                    Recovered target points.
        """

        B, H, W = src_mask_b.shape

        # =================== source mask preprocessing ================

        src_mask_b = (torch.from_numpy(src_mask_b).unsqueeze(1).float().to(
            feat_map_b.device))  # [B, 1, H, W]

        # Convert binary masks into strong logits
        src_mask_b = torch.where(src_mask_b > 0.5, 30.0, -30.0)

        # ================= target refinement mask ================

        if tar_mask_b is not None:

            tar_mask_b = F.interpolate(
                tar_mask_b.float(),
                size=(self.prompt_encoder.mask_input_size,
                      self.prompt_encoder.mask_input_size),
                align_corners=False,
                mode="bilinear",
                antialias=True,
            )

        # ======================== prompt encoding ========================

        (
            sparse_embeddings_b,
            src_dense_embeddings_img_b,
            tar_dense_embeddings_img_b,
        ) = self.prompt_encoder(
            src_points_b=src_points_b,
            tar_points_b=tar_points_b,
            src_mask_b=src_mask_b,
            tar_mask_b=tar_mask_b,
            src_points_feat_b=src_points_feat_b,
        )

        # ======================== mask decoding ========================

        low_res_mask_b = self.mask_decoder(
            feat_map_b=feat_map_b,

            # Cross-attention positional encoding
            ca_img_pe=self.prompt_encoder.get_dense_ca_pe(),

            # Self-attention positional encoding
            sa_img_pe=self.prompt_encoder.get_dense_sa_pe(),
            sparse_embeddings_b=sparse_embeddings_b,
            src_dense_embeddings_img_b=src_dense_embeddings_img_b,
            tar_dense_embeddings_img_b=tar_dense_embeddings_img_b,
        )

        # ======================== mask upsampling ========================

        pred_mask_b = F.interpolate(low_res_mask_b,
                                    size=(H, W),
                                    mode='bilinear',
                                    align_corners=False)  # [B, 1, H, W]

        # Clamp logits for numerical stability
        low_res_mask_b = torch.clamp(low_res_mask_b, -32.0, 32.0)

        # ======================== training mode ========================

        if ret_logits:
            return pred_mask_b, low_res_mask_b

        # ======================== recovery mode ========================

        pred_mask_recover_b = []
        pred_points_recover_b = []

        for b in range(B):

            # Recover cropped predictions
            if meta[b] is not None:

                pred_mask_recover, pred_points_recover = (
                    recover_square_crop_tensor(pred_mask_b[b], tar_points_b[b],
                                               meta[b]))

                pred_mask_recover = (pred_mask_recover > 0.0)

            # Direct resize recovery
            else:

                pred_mask_recover = F.interpolate(pred_mask_b[b].unsqueeze(0),
                                                  size=(tar_square,
                                                        tar_square),
                                                  mode='bilinear',
                                                  align_corners=False)[0,
                                                                       0] > 0.0

                pred_points_recover = (tar_points_b[b] / H * 704)

            pred_mask_recover_b.append(pred_mask_recover)

            pred_points_recover_b.append(pred_points_recover)

        # Convert logits into binary masks
        pred_mask_b = pred_mask_b > 0.0

        return (pred_mask_b, low_res_mask_b, pred_mask_recover_b,
                pred_points_recover_b)

    def _load_resume(self, ckpt_path):
        """
        Load pretrained checkpoint.

        Supports:
            - single GPU checkpoints
            - DistributedDataParallel checkpoints

        Args:
            ckpt_path (str):
                Checkpoint file path.
        """

        ckpt = torch.load(ckpt_path, map_location='cpu')

        tar_state = ckpt['model']

        new_state = OrderedDict()

        for k, v in tar_state.items():

            # Remove DDP prefix
            if k.startswith('module.'):
                new_k = k[len('module.'):]
            else:
                new_k = k

            # Only load matched parameters
            if new_k in self.state_dict().keys():
                new_state[new_k] = v

        # Ensure all parameters are loaded
        assert (len(self.state_dict().keys()) == len(new_state.keys()))

        self.load_state_dict(new_state)

        del ckpt

        print(f'[NOTE] Resume from {ckpt_path}')
