"""
Mobile config for SHARP on Android (ExecuTorch / LiteRT pipeline).

Matches ExecutorchSharp.kt and export_sharp_executorch_split4.py:
- Image 1536x1536, patch 384x384, pyramid 1x / 0.5x / 0.25x.
- Grid: 5×5 at 1x (25), 3×3 at 0.5x (9), 1 at 0.25x = 35 patches.
- Merge padding: 3 for 1x, 6 for 0.5x; spatial 24 per patch after ViT.
"""

IMAGE_SIZE = 1536
PATCH_SIZE = 384
SPATIAL_SIZE = 24  # 384 / 16
FEATURE_DIM = 1024

# Pyramid grid (Urdhva-tiryagbhyam: vertical and crosswise)
GRID_1X = 5
GRID_05X = 3
PATCHES_1X = 25
PATCHES_05X = 9
PATCHES_025X = 1
TOTAL_PATCHES = 35

# Overlap ratio for patch extraction (same as export script)
OVERLAP_1X = 0.25
OVERLAP_05X = 0.25

# Merge padding (NCNN/ExecutorchSharp)
PADDING_1X = 3
PADDING_05X = 6

# Token/feature shapes
IMAGE_TOKENS_SEQ_LEN = 577
GAUSSIAN_CHANNELS = 14


def get_merged_size(grid_size: int, patch_spatial: int = SPATIAL_SIZE, padding: int = 0) -> int:
    """Merged feature map size for a grid of patches with overlap (Nikhilam-style contrib)."""
    patch_contrib = patch_spatial - 2 * padding
    return patch_spatial + (grid_size - 1) * patch_contrib


# Expected merge sizes for mobile pipeline
MERGED_SIZE_1X = get_merged_size(GRID_1X, SPATIAL_SIZE, PADDING_1X)   # 96
MERGED_SIZE_05X = get_merged_size(GRID_05X, SPATIAL_SIZE, PADDING_05X)  # 48
MERGED_SIZE_025X = SPATIAL_SIZE  # 24
