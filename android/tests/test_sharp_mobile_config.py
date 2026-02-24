"""
Test SHARP mobile config with a picked image.

- Picks or creates a test image (1536x1536 RGB).
- Uses mobile config (image size, patch size, grid, merge sizes).
- Asserts patch count and tensor shapes match the Android pipeline.
"""

import math
import os
import sys
from pathlib import Path

import pytest

# Ensure android/ is on path when run from repo root
ANDROID_DIR = Path(__file__).resolve().parent.parent
if str(ANDROID_DIR) not in sys.path:
    sys.path.insert(0, str(ANDROID_DIR))

# Import from sibling module (tests run with android/tests as cwd or path)
from mobile_config import (
    IMAGE_SIZE,
    PATCH_SIZE,
    SPATIAL_SIZE,
    GRID_1X,
    GRID_05X,
    TOTAL_PATCHES,
    PADDING_1X,
    PADDING_05X,
    MERGED_SIZE_1X,
    MERGED_SIZE_05X,
    MERGED_SIZE_025X,
    FEATURE_DIM,
    get_merged_size,
)

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
TEST_IMAGE_PATH = FIXTURES_DIR / "test_room_1536.png"


def _create_test_image(width: int, height: int, path: Path) -> Path:
    """Create a simple RGB test image and save it. Uses PIL if available else skip."""
    if not PIL_AVAILABLE:
        pytest.skip("PIL not available to create test image")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Gradient + a bit of structure so it's not blank
    img = Image.new("RGB", (width, height))
    pix = img.load()
    for y in range(height):
        for x in range(width):
            r = int(255 * (x / width))
            g = int(255 * (y / height))
            b = 128
            pix[x, y] = (r, g, b)
    img.save(path)
    return path


def _load_image_as_tensor(path: Path) -> "torch.Tensor":
    """Load image from path to NCHW float tensor [1, 3, H, W] in [0,1]."""
    if not PIL_AVAILABLE or not TORCH_AVAILABLE:
        pytest.skip("PIL and torch required to load image")
    np = pytest.importorskip("numpy")
    img = Image.open(path).convert("RGB")
    arr = torch.from_numpy(np.array(img)).float() / 255.0
    # HWC -> NCHW
    arr = arr.permute(2, 0, 1).unsqueeze(0)
    return arr


def split_patches_list(image: "torch.Tensor", overlap_ratio: float, patch_size: int):
    """Same logic as export_sharp_executorch_split4.split_patches_list."""
    patch_stride = int(patch_size * (1 - overlap_ratio))
    image_size = image.shape[-1]
    steps = int(math.ceil((image_size - patch_size) / patch_stride)) + 1
    patches = []
    for j in range(steps):
        j0 = j * patch_stride
        for i in range(steps):
            i0 = i * patch_stride
            patches.append(image[..., j0 : j0 + patch_size, i0 : i0 + patch_size])
    return patches


@pytest.fixture(scope="module")
def test_image_path():
    """Pick image: use fixture if present, else create a 1536x1536 test image."""
    if TEST_IMAGE_PATH.exists():
        return TEST_IMAGE_PATH
    if PIL_AVAILABLE:
        return _create_test_image(IMAGE_SIZE, IMAGE_SIZE, TEST_IMAGE_PATH)
    pytest.skip("PIL required to create or load test image")


@pytest.fixture(scope="module")
def image_tensor(test_image_path):
    """Load test image as [1, 3, 1536, 1536] float tensor."""
    arr = _load_image_as_tensor(test_image_path)
    # Resize to exact mobile size if needed
    if arr.shape[2] != IMAGE_SIZE or arr.shape[3] != IMAGE_SIZE:
        arr = F.interpolate(
            arr, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
        )
    return arr


@pytest.fixture
def mobile_config():
    """Mobile config dict for assertions."""
    return {
        "image_size": IMAGE_SIZE,
        "patch_size": PATCH_SIZE,
        "spatial_size": SPATIAL_SIZE,
        "grid_1x": GRID_1X,
        "grid_05x": GRID_05X,
        "total_patches": TOTAL_PATCHES,
        "merged_size_1x": MERGED_SIZE_1X,
        "merged_size_05x": MERGED_SIZE_05X,
        "merged_size_025x": MERGED_SIZE_025X,
        "feature_dim": FEATURE_DIM,
    }


# --- Tests ---


def test_mobile_config_constants(mobile_config):
    """Mobile config matches Android (ExecutorchSharp / export script)."""
    assert mobile_config["image_size"] == 1536
    assert mobile_config["patch_size"] == 384
    assert mobile_config["total_patches"] == 35
    assert mobile_config["merged_size_1x"] == 96
    assert mobile_config["merged_size_05x"] == 48
    assert mobile_config["merged_size_025x"] == 24
    assert get_merged_size(GRID_1X, SPATIAL_SIZE, PADDING_1X) == 96
    assert get_merged_size(GRID_05X, SPATIAL_SIZE, PADDING_05X) == 48


def test_image_shape(image_tensor, mobile_config):
    """Picked image has mobile config shape [1, 3, 1536, 1536]."""
    assert image_tensor.dim() == 4
    assert image_tensor.shape[0] == 1
    assert image_tensor.shape[1] == 3
    assert image_tensor.shape[2] == mobile_config["image_size"]
    assert image_tensor.shape[3] == mobile_config["image_size"]


def test_patch_extraction_counts(image_tensor, mobile_config):
    """Patch extraction with mobile config yields 25 + 9 + 1 = 35 patches."""
    x0 = image_tensor
    x1 = F.interpolate(x0, scale_factor=0.5, mode="bilinear", align_corners=False)
    x2 = F.interpolate(x0, scale_factor=0.25, mode="bilinear", align_corners=False)

    patches_1x = split_patches_list(x0, 0.25, PATCH_SIZE)
    patches_05x = split_patches_list(x1, 0.25, PATCH_SIZE)

    assert len(patches_1x) == 25, "1x scale: 5×5 grid"
    assert len(patches_05x) == 9, "0.5x scale: 3×3 grid"
    total = len(patches_1x) + len(patches_05x) + 1
    assert total == mobile_config["total_patches"]


def test_patch_shapes(image_tensor):
    """1x patches are [1, 3, 384, 384]; 0.5x count is 9 (edge patches may be smaller on 768px)."""
    x0 = image_tensor
    x1 = F.interpolate(x0, scale_factor=0.5, mode="bilinear", align_corners=False)
    x2 = F.interpolate(x0, scale_factor=0.25, mode="bilinear", align_corners=False)

    patches_1x = split_patches_list(x0, 0.25, PATCH_SIZE)
    patches_05x = split_patches_list(x1, 0.25, PATCH_SIZE)

    for p in patches_1x:
        assert p.shape == (1, 3, PATCH_SIZE, PATCH_SIZE), "1x patches must be 384x384"
    assert len(patches_05x) == 9
    for p in patches_05x:
        assert p.dim() == 4 and p.shape[0] == 1 and p.shape[1] == 3
        assert p.shape[2] <= PATCH_SIZE and p.shape[3] <= PATCH_SIZE
    assert x2.shape == (1, 3, PATCH_SIZE, PATCH_SIZE)


def test_merge_sizes_from_config():
    """Merge size formula matches Android getMergedSize (Nikhilam-style contrib)."""
    assert get_merged_size(5, 24, 3) == 96
    assert get_merged_size(3, 24, 6) == 48
    assert get_merged_size(1, 24, 0) == 24


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
