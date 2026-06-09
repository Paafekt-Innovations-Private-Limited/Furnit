#!/usr/bin/env python3
"""Export RTMDet-Ins-m raw head tensors to Core ML.

This bypasses MMDeploy post-processing. The Core ML graph ends at the last
convolutional outputs:
  cls_80/40/20, bbox_80/40/20, kernel_80/40/20, mask_feat

Swift owns decode, NMS, kernel selection, relative coords, dynamic mask MLP,
sigmoid, crop, and threshold.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import coremltools as ct
import torch
from mmdet.apis import init_detector


MMDET_DIR = Path("/opt/openmmlab/mmdetection")
MODEL_CONFIG = MMDET_DIR / "configs/rtmdet/rtmdet-ins_m_8xb32-300e_coco.py"
WORK_DIR = Path("/work/.artifacts/rtmdet-coreml-raw-export")
IOS_DROP_DIR = Path("/work/Furnit/Models/RTMDet")
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
    parser.add_argument("--output-name", default="rtmdet-ins-m.mlpackage")
    args = parser.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("RTMDET_TORCH_THREADS", "1")))
    torch.set_num_interop_threads(1)
    print(">>> loading RTMDet checkpoint", flush=True)
    model = init_detector(str(MODEL_CONFIG), str(checkpoint_path()), device="cpu")
    model.eval()

    wrapped = RTMDetInsRawHead(model).eval()
    example = torch.zeros(1, 3, 640, 640, dtype=torch.float32)
    with torch.inference_mode():
        print(">>> probing backbone output shapes", flush=True)
        start = time.perf_counter()
        feats = model.extract_feat(example)
        print(f">>> backbone probe took {time.perf_counter() - start:.2f}s", flush=True)
        for idx, feat in enumerate(feats):
            print(f"feat_{idx}: {tuple(feat.shape)}", flush=True)

        print(">>> probing bbox/mask head output shapes", flush=True)
        start = time.perf_counter()
        outputs = model.bbox_head(feats)
        print(f">>> bbox/mask head probe took {time.perf_counter() - start:.2f}s", flush=True)
        outputs = (
            outputs[0][0],
            outputs[0][1],
            outputs[0][2],
            outputs[1][0],
            outputs[1][1],
            outputs[1][2],
            outputs[2][0],
            outputs[2][1],
            outputs[2][2],
            outputs[3],
        )

        print(">>> probing wrapped raw model", flush=True)
        start = time.perf_counter()
        _ = wrapped(example)
        print(f">>> wrapped raw probe took {time.perf_counter() - start:.2f}s", flush=True)

        print(">>> tracing raw head", flush=True)
        start = time.perf_counter()
        traced = torch.jit.trace(wrapped, example, strict=False, check_trace=False)
        print(f">>> trace took {time.perf_counter() - start:.2f}s", flush=True)
        print(">>> validating traced raw head output shapes", flush=True)
        start = time.perf_counter()
        outputs = traced(example)
        print(f">>> traced validation took {time.perf_counter() - start:.2f}s", flush=True)

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
    for name, tensor in zip(output_names, outputs):
        print(f"{name}: {tuple(tensor.shape)}", flush=True)

    package = WORK_DIR / "rtmdet-ins-m-raw.mlpackage"
    if package.exists():
        shutil.rmtree(package)

    print(">>> converting raw head to Core ML", flush=True)
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="input", shape=example.shape)],
        outputs=[ct.TensorType(name=name) for name in output_names],
        minimum_deployment_target=ct.target.iOS17,
    )
    print(">>> saving raw Core ML package", flush=True)
    mlmodel.save(str(package))

    IOS_DROP_DIR.mkdir(parents=True, exist_ok=True)
    destination = IOS_DROP_DIR / args.output_name
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(package, destination)
    print(f"Raw Core ML package: {package}")
    print(f"Installed for iOS: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
