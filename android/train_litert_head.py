#!/usr/bin/env python3
"""
Train a pure GaussianHead from the original SHARP model.
NO NCNN weights. Uses only sharp_2572gikvuh.pt as teacher.

Pipeline:
  1. Load full SHARP predictor
  2. For N real images + augmentations: run full model -> ground truth gaussians
  3. For same images: run patch encoder -> merge -> [1024, 96, 96] features
  4. Train GaussianHead to map features -> gaussian params
  5. Export to TFLite via litert-torch
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

SHARP_SRC = Path("/tmp/ml-sharp/src")
sys.path.insert(0, str(SHARP_SRC))

MODEL_WEIGHTS = Path(
    "/Users/al/Library/Mobile Documents/com~apple~CloudDocs/"
    "ml_experiments/models/sharp_2572gikvuh.pt"
)
OUTPUT_DIR = Path("/Users/al/Documents/tries01/Furnit/android/sharp_ncnn_models")

# Real room/interior images
IMAGE_PATHS = [
    Path("/Users/al/Documents/tries01/Furnit/android/room.jpeg"),
    Path("/Users/al/Documents/tries01/Furnit/android/TestRoom.jpg"),
    Path("/Users/al/Documents/tries01/Furnit/FurnitTests/landscape.jpeg"),
    Path("/Users/al/Documents/tries01/Furnit/android/app/src/main/assets/room_previews/vintage.jpg"),
    Path("/tmp/ml-sharp/data/teaser.jpg"),
]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class SinglePatchEncoder(nn.Module):
    """ViT patch encoder - same as export_sharp_litert.py."""

    def __init__(self, vit):
        super().__init__()
        self.patch_embed = vit.patch_embed
        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed
        self.blocks = vit.blocks
        self.norm = vit.norm

    @torch.no_grad()
    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed.proj(patch)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(patch.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x[:, 1:, :]
        B = x.shape[0]
        return x.transpose(1, 2).reshape(B, 1024, 24, 24)


class GaussianHead(nn.Module):
    """Lightweight conv head: [1024,96,96] -> [14,384,384]."""

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(1024, 256, 1, bias=True)
        self.conv1 = nn.Conv2d(256, 256, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(256, 128, 3, padding=1, bias=True)
        self.conv4 = nn.Conv2d(128, 64, 3, padding=1, bias=True)
        self.conv5 = nn.Conv2d(64, 14, 1, bias=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv0(x))
        x = self.relu(self.conv1(x))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.relu(self.conv2(x))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image(path: Path, size=1536) -> torch.Tensor:
    """Load an image, resize to size x size, return [1, 3, size, size] in [0, 1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    tensor = T.ToTensor()(img).unsqueeze(0)  # [1, 3, H, W] in [0, 1]
    return tensor


def augment_image(img_tensor: torch.Tensor, idx: int) -> torch.Tensor:
    """Apply deterministic augmentation based on idx.
    img_tensor: [1, 3, H, W] in [0, 1]
    Returns: [1, 3, H, W] in [0, 1]
    """
    x = img_tensor.clone()
    if idx == 0:
        return x  # original
    if idx == 1:
        return TF.hflip(x)  # horizontal flip
    if idx == 2:
        # slight brightness increase
        return (x * 1.15).clamp(0, 1)
    if idx == 3:
        # slight brightness decrease
        return (x * 0.85).clamp(0, 1)
    if idx == 4:
        # hflip + brightness
        return (TF.hflip(x) * 1.1).clamp(0, 1)
    return x


def merge_1x(patch_feats, grid=5, pad=3):
    """Merge 25 x [1,1024,24,24] -> [1,1024,96,96]."""
    S = 24
    contrib = S - 2 * pad
    out = S + (grid - 1) * contrib          # 96
    merged = torch.zeros(1, 1024, out, out, device=patch_feats[0].device)

    idx, oy = 0, 0
    for j in range(grid):
        y0 = 0 if j == 0 else pad
        y1 = S if j == grid - 1 else S - pad
        h = y1 - y0
        ox = 0
        for i in range(grid):
            x0 = 0 if i == 0 else pad
            x1 = S if i == grid - 1 else S - pad
            w = x1 - x0
            merged[:, :, oy:oy + h, ox:ox + w] = patch_feats[idx][:1, :, y0:y1, x0:x1]
            ox += w
            idx += 1
        oy += h
    return merged


def encode_patches(encoder, img):
    """Run patch encoder on 25 1x patches, merge to [1,1024,96,96]."""
    stride = 288
    feats = []
    for r in range(5):
        for c in range(5):
            patch = img[:, :, r * stride:r * stride + 384,
                              c * stride:c * stride + 384]
            feats.append(encoder(patch).cpu())
    return merge_1x(feats)


def gaussians_to_target(g, sz=384):
    """Gaussians3D -> [1, 14, sz, sz] training target.

    Channel layout (matching extractGaussians on Android):
      0-2   position xyz
      3     opacity  (logit, i.e. inverse-sigmoid)
      4-6   scale    (final values, positive)
      7-10  quaternion wxyz (un-normalised is fine)
      11-13 color rgb (0-1)
    """
    B, N = 1, g.mean_vectors.shape[1]
    L, H, W = 2, 768, 768
    assert N == L * H * W, f"expected {L * H * W}, got {N}"

    pos   = g.mean_vectors.reshape(B, L, H, W, 3).permute(0, 4, 1, 2, 3)    # [1,3,2,H,W]
    sv    = g.singular_values.reshape(B, L, H, W, 3).permute(0, 4, 1, 2, 3)  # [1,3,2,H,W]
    quat  = g.quaternions.reshape(B, L, H, W, 4).permute(0, 4, 1, 2, 3)      # [1,4,2,H,W]
    col   = g.colors.reshape(B, L, H, W, 3).permute(0, 4, 1, 2, 3)           # [1,3,2,H,W]
    opac  = g.opacities.reshape(B, L, H, W)                                   # [1,2,H,W]

    # First layer only
    pos0  = pos[:, :, 0]                                  # [1,3,H,W]
    sv0   = sv[:, :, 0]
    q0    = quat[:, :, 0]
    c0    = col[:, :, 0]
    o0    = opac[:, 0:1]                                   # [1,1,H,W]

    # Inverse-sigmoid for opacity so Android can apply sigmoid later
    o0_logit = torch.log(o0.clamp(1e-4, 1 - 1e-4) / (1 - o0.clamp(1e-4, 1 - 1e-4)))

    target = torch.cat([pos0, o0_logit, sv0, q0, c0], dim=1)   # [1,14,768,768]
    return F.interpolate(target, size=(sz, sz), mode="bilinear", align_corners=False)


def channel_weighted_loss(pred, target):
    """MSE loss with per-channel-group weighting.
    Channels: pos(0-2), opacity(3), scale(4-6), quat(7-10), color(11-13)
    Color channels weighted higher to prevent rainbow artifacts.
    """
    weights = torch.ones(14, device=pred.device)
    weights[0:3]   = 1.0   # position
    weights[3]     = 2.0   # opacity (important for visibility)
    weights[4:7]   = 1.0   # scale
    weights[7:11]  = 1.0   # quaternion
    weights[11:14] = 3.0   # color (weighted heavily to fix rainbow)

    diff_sq = (pred - target) ** 2  # [B, 14, H, W]
    weighted = diff_sq * weights.view(1, 14, 1, 1)
    return weighted.mean()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 60)
    print("Train Pure GaussianHead (real images, original SHARP only)")
    print("=" * 60)

    if not SHARP_SRC.exists():
        print(f"ERROR: {SHARP_SRC} not found"); return 1
    if not MODEL_WEIGHTS.exists():
        print(f"ERROR: {MODEL_WEIGHTS} not found"); return 1

    # Check which images exist
    valid_images = [p for p in IMAGE_PATHS if p.exists()]
    print(f"\nFound {len(valid_images)} images:")
    for p in valid_images:
        print(f"  {p.name}")
    if not valid_images:
        print("ERROR: No images found!"); return 1

    # ---- Load full SHARP predictor ----
    from sharp.models import PredictorParams, create_predictor

    print("\nLoading full SHARP model ...")
    sd = torch.load(MODEL_WEIGHTS, map_location="cpu", weights_only=False)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(sd)
    predictor.eval()
    del sd
    print(f"  Predictor loaded ({sum(p.numel() for p in predictor.parameters()) / 1e6:.0f}M params)")

    # ---- Encoder (same weights the TFLite patch encoder uses) ----
    vit = predictor.monodepth_model.monodepth_predictor.encoder.patch_encoder
    encoder = SinglePatchEncoder(vit)
    encoder.eval()

    # ---- Gaussian head (random init, will be trained) ----
    head = GaussianHead()
    print(f"  GaussianHead: {sum(p.numel() for p in head.parameters()) / 1e6:.1f}M params")

    # ---- Generate training pairs from real images + augmentations ----
    NUM_AUGS = 3  # original + 2 augmentations per image (memory-friendly)
    EPOCHS = 500

    pairs = []
    total_samples = len(valid_images) * NUM_AUGS
    print(f"\nGenerating {total_samples} training pairs ({len(valid_images)} images x {NUM_AUGS} augmentations) ...")

    import gc
    sample_idx = 0
    for img_path in valid_images:
        base_img = load_image(img_path, size=1536)
        print(f"\n  Image: {img_path.name} ({base_img.shape})")

        for aug_idx in range(NUM_AUGS):
            sample_idx += 1
            img = augment_image(base_img, aug_idx)
            aug_name = ["original", "hflip", "bright+", "bright-", "hflip+bright"][aug_idx]

            # Features via patch encoder (same as LiteRT on-device)
            merged = encode_patches(encoder, img)

            # Ground truth via full model
            disp_f = torch.tensor([1.0])
            print(f"    [{sample_idx}/{total_samples}] {aug_name} ... ", end="", flush=True)
            ts = time.time()
            with torch.no_grad():
                gaussians = predictor(img, disp_f)
            target = gaussians_to_target(gaussians)
            elapsed = time.time() - ts
            print(f"{elapsed:.0f}s  pos[{target[0, :3].min():.2f},{target[0, :3].max():.2f}] "
                  f"col[{target[0, 11:14].min():.2f},{target[0, 11:14].max():.2f}]")

            pairs.append((merged.detach(), target.detach()))
            del img, merged, gaussians, target
            gc.collect()

        del base_img
        gc.collect()

    # Free the full predictor before training - saves ~3GB
    del predictor, encoder, vit
    gc.collect()
    print(f"\nTotal training pairs: {len(pairs)}  (freed predictor, ~3GB saved)")

    # ---- Train ----
    print(f"\nTraining for {EPOCHS} epochs ...")
    optim = torch.optim.Adam(head.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)

    best_loss = float("inf")
    best_state = None

    head.train()
    for ep in range(EPOCHS):
        total = 0.0
        for feat, tgt in pairs:
            pred = head(feat)
            loss = channel_weighted_loss(pred, tgt)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item()
        sched.step()
        avg_loss = total / len(pairs)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

        if (ep + 1) % 50 == 0 or ep == 0:
            print(f"  epoch {ep + 1:4d}  loss={avg_loss:.6f}  best={best_loss:.6f}  lr={sched.get_last_lr()[0]:.2e}")

    # Load best weights
    head.load_state_dict(best_state)
    print(f"\n  Best loss: {best_loss:.6f}")

    # ---- Validate: check per-channel error on training data ----
    head.eval()
    print("\nPer-channel MSE on training data:")
    ch_names = ["pos_x", "pos_y", "pos_z", "opacity", "scale_x", "scale_y", "scale_z",
                "quat_w", "quat_x", "quat_y", "quat_z", "color_r", "color_g", "color_b"]
    ch_errors = torch.zeros(14)
    with torch.no_grad():
        for feat, tgt in pairs:
            pred = head(feat)
            for c in range(14):
                ch_errors[c] += F.mse_loss(pred[:, c], tgt[:, c]).item()
    ch_errors /= len(pairs)
    for c in range(14):
        print(f"  {ch_names[c]:10s}: {ch_errors[c]:.6f}")

    # ---- Export ----
    print("\nExporting to TFLite ...")
    import litert_torch

    dummy = torch.randn(1, 1024, 96, 96)
    with torch.no_grad():
        ref = head(dummy)

    edge = litert_torch.convert(head, (dummy,))
    out_path = OUTPUT_DIR / "sharp_gaussian_head.tflite"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    edge.export(str(out_path))

    edge_out = edge(dummy)
    diff = (ref - edge_out).abs().mean().item()
    sz = out_path.stat().st_size / 1024 / 1024

    print(f"  {out_path.name}  ({sz:.1f} MB)  conversion err={diff:.6f}")
    print(f"\nTotal time: {time.time() - t0:.0f}s")
    print(f"\nadb push {out_path} /data/local/tmp/furnit/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
