#!/usr/bin/env python3
"""Export RTMDet-Ins-m raw head tensors to ONNX for Android.

The exported graph matches the iOS raw Core ML contract:
  cls_80/40/20, bbox_80/40/20, kernel_80/40/20, mask_feat

Android owns decode, NMS, kernel selection, dynamic mask MLP, crop, and alpha
matte rendering.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from mmdet.apis import init_detector


MMDET_DIR = Path("/opt/openmmlab/mmdetection")
MODEL_CONFIG = MMDET_DIR / "configs/rtmdet/rtmdet-ins_m_8xb32-300e_coco.py"
WORK_DIR = Path("/work/.artifacts/rtmdet-android-raw-export")
ANDROID_ASSETS_DIR = Path("/work/android/app/src/main/assets")
CHECKPOINT_NAME = "rtmdet-ins_m_8xb32-300e_coco_20221123_001039-6eba602e.pth"


class RTMDetInsRawHead(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        feats = self.model.extract_feat(x)
        cls_scores, bbox_preds, kernel_preds, mask_feat = self.model.bbox_head(feats)
        return (
            cls_scores[0],
            cls_scores[1],
            cls_scores[2],
            bbox_preds[0],
            bbox_preds[1],
            bbox_preds[2],
            kernel_preds[0],
            kernel_preds[1],
            kernel_preds[2],
            mask_feat,
        )


def checkpoint_path() -> Path:
    env = os.environ.get("RTMDET_CKPT")
    if env:
        path = Path(env)
        if path.is_file() and path.stat().st_size > 0:
            return path
    cached = Path("/work/.modelcache") / CHECKPOINT_NAME
    if cached.is_file() and cached.stat().st_size > 0:
        return cached
    raise FileNotFoundError(f"Missing checkpoint: {cached}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-name", default="rtmdet-ins-m-raw.onnx")
    args = parser.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("RTMDET_TORCH_THREADS", "1")))
    torch.set_num_interop_threads(1)

    print(">>> loading RTMDet checkpoint", flush=True)
    model = init_detector(str(MODEL_CONFIG), str(checkpoint_path()), device="cpu")
    model.eval()
    wrapped = RTMDetInsRawHead(model).eval()
    example = torch.zeros(1, 3, 640, 640, dtype=torch.float32)

    output_names = [
        "cls_80",
        "cls_40",
        "cls_20",
        "bbox_80",
        "bbox_40",
        "bbox_20",
        "kernel_80",
        "kernel_40",
        "kernel_20",
        "mask_feat",
    ]

    with torch.inference_mode():
        print(">>> probing raw output shapes", flush=True)
        outputs = wrapped(example)
        for name, tensor in zip(output_names, outputs):
            print(f"{name}: {tuple(tensor.shape)}", flush=True)

        onnx_path = WORK_DIR / args.output_name
        if onnx_path.exists():
            onnx_path.unlink()

        print(f">>> exporting ONNX: {onnx_path}", flush=True)
        torch.onnx.export(
            wrapped,
            example,
            str(onnx_path),
            input_names=["input"],
            output_names=output_names,
            opset_version=17,
            do_constant_folding=True,
        )

    ANDROID_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    android_path = ANDROID_ASSETS_DIR / args.output_name
    android_path.write_bytes(onnx_path.read_bytes())
    print(f"Raw ONNX: {onnx_path}")
    print(f"Installed for Android: {android_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
