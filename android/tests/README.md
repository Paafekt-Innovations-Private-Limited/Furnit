# Android SHARP / ExecuTorch tests

## Test: mobile config + image

- **Config:** `mobile_config.py` — Android pipeline constants (1536×1536, 384 patch, 5×5+3×3+1 = 35 patches, merge sizes 96/48/24).
- **Image:** Picked image for tests: use `tests/fixtures/test_room_1536.png` if present; otherwise the test creates a 1536×1536 RGB gradient in that path.
- **Run:** From `android/`:
  ```bash
  python -m pytest tests/test_sharp_mobile_config.py -v
  ```
  Or with the script directly:
  ```bash
  python tests/test_sharp_mobile_config.py
  ```

## Requirements

- `pytest`
- `torch` (for image tensor and patch extraction tests)
- `PIL` (Pillow) and `numpy` (to load/create the test image)

Tests that only assert config constants and merge-size formula run without torch/PIL.
