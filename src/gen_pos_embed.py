"""
Adapted from https://github.com/facebookresearch/vggt
# Precompute and save positional embeddings offline
# for faster loading and initialization.
"""

import argparse
from pathlib import Path
from typing import Tuple, Union

import torch

SAVE_DIR = Path(__file__).parent / "../third_party/vggt_main/pos_embed"
SAVE_DIR = SAVE_DIR.resolve()


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: Union[int, Tuple[int, int]],
    return_grid: bool = False,
) -> torch.Tensor:
    """
    Initialize a grid and generate a 2D sine-cosine positional embedding.

    Args:
        embed_dim: Embedding dimension.
        grid_size: Grid size (int or tuple of H, W).
        return_grid: Whether to also return the coordinate grid.

    Returns:
        pos_embed: Shape [1, C, H, W]
    """
    if isinstance(grid_size, tuple):
        grid_size_h, grid_size_w = grid_size
    else:
        grid_size_h = grid_size_w = grid_size

    grid_h = torch.arange(grid_size_h, dtype=torch.float)
    grid_w = torch.arange(grid_size_w, dtype=torch.float)

    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    pos_embed = (pos_embed.reshape(1, grid_size_h, grid_size_w,
                                   -1).permute(0, 3, 1, 2))

    if return_grid:
        return pos_embed, grid

    return pos_embed


def get_2d_sincos_pos_embed_from_grid(
    embed_dim: int,
    grid: torch.Tensor,
) -> torch.Tensor:
    """
    Generate 2D sine-cosine positional embedding from a grid.

    Args:
        embed_dim: Embedding dimension.
        grid: Coordinate grid.

    Returns:
        emb: Positional embedding.
    """
    assert embed_dim % 2 == 0

    # Half dimensions for H and W respectively
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = torch.cat([emb_h, emb_w], dim=2)
    return emb


def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int,
    pos: torch.Tensor,
) -> torch.Tensor:
    """
    Generate 1D sine-cosine positional embedding.

    Args:
        embed_dim: Embedding dimension.
        pos: Position tensor.

    Returns:
        emb: Positional embedding.
    """
    assert embed_dim % 2 == 0

    omega = torch.arange(embed_dim // 2, dtype=torch.double)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)

    pos = pos.reshape(-1)

    out = torch.einsum("m,d->md", pos, omega)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=1)

    return emb[None].float()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--img_H",
        type=int,
        default=518,
        help="Input image height.",
    )

    parser.add_argument(
        "--img_W",
        type=int,
        default=518,
        help="Input image width.",
    )

    parser.add_argument(
        "--dim",
        type=int,
        default=388,
        help=("Positional embedding dimension. "
              "It is not recommended to modify this value, "
              "as it corresponds to VGGT's default embedding dimension."),
    )

    args = parser.parse_args()

    img_H = args.img_H
    img_W = args.img_W
    dim = args.dim

    grid_H = img_H // 2
    grid_W = img_W // 2

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    save_path = SAVE_DIR / f"pos_embed_{dim}_{grid_H}_{grid_W}.pt"

    pos_embed = get_2d_sincos_pos_embed(
        dim,
        grid_size=(grid_H, grid_W),
    )

    torch.save(pos_embed, save_path)

    loaded = torch.load(save_path)

    print(f"Saved to: {save_path}")
    print(f"Shape: {loaded.shape}")

    print(
        "Equal after load:",
        torch.equal(
            loaded,
            get_2d_sincos_pos_embed(
                dim,
                grid_size=(grid_H, grid_W),
            ),
        ),
    )


if __name__ == "__main__":
    main()
