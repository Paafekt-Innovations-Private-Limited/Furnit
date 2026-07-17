#!/usr/bin/env python3
"""
Export RTMDet-Ins-m to Core ML from a Linux container environment.

Expected environment:
- Python 3.10
- torch==2.1.2 / torchvision==0.16.2
- mmcv==2.1.0
- mmengine==0.10.3
- mmdet==3.3.0
- mmdeploy==1.3.1
- coremltools==7.1

This script writes the custom MMDeploy config needed for instance segmentation
with a Core ML backend, downloads the official RTMDet-Ins-m checkpoint if
missing, runs MMDeploy conversion, and copies the resulting `.mlpackage`
into the iOS app drop zone.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MMDEPLOY_DIR = Path("/opt/openmmlab/mmdeploy")
MMDET_DIR = Path("/opt/openmmlab/mmdetection")
CHECKPOINTS_DIR = Path("/work/.artifacts/rtmdet-checkpoints")
WORK_DIR = Path("/work/.artifacts/rtmdet-coreml-export")
IOS_DROP_DIR = Path("/work/Furnit/Models/RTMDet")
MODEL_CONFIG = MMDET_DIR / "configs/rtmdet/rtmdet-ins_m_8xb32-300e_coco.py"
CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
    "rtmdet-ins_m_8xb32-300e_coco/"
    "rtmdet-ins_m_8xb32-300e_coco_20221123_001039-6eba602e.pth"
)
CHECKPOINT_PATH = CHECKPOINTS_DIR / "rtmdet-ins_m_8xb32-300e_coco_20221123_001039-6eba602e.pth"
TEST_IMAGE = Path("/work/bus.jpg")


DEPLOY_CONFIG_TEXT = """_base_ = [
    '../_base_/base_instance-seg_torchscript.py',
    '../../_base_/backends/coreml.py',
]

ir_config = dict(
    input_shape=(640, 640),
    output_names=['dets', 'labels', 'masks'],
)

backend_config = dict(
    type='coreml',
    convert_to='mlprogram',
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 640, 640],
                    max_shape=[1, 3, 640, 640],
                    default_shape=[1, 3, 640, 640],
                ))),
    ],
)

codebase_config = dict(
    type='mmdet',
    model_type='end2end',
    post_processing=dict(
        score_threshold=0.05,
        confidence_threshold=0.005,
        iou_threshold=0.5,
        max_output_boxes_per_class=200,
        pre_top_k=5000,
        keep_top_k=100,
        background_label_id=-1,
        export_postprocess_mask=True,
    ))
"""


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_deploy_config() -> Path:
    path = MMDEPLOY_DIR / "configs/mmdet/instance-seg/instance-seg_rtmdet-ins_coreml_static-640x640.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEPLOY_CONFIG_TEXT, encoding="utf-8")
    return path


def ensure_checkpoint() -> Path:
    env_ckpt = os.environ.get("RTMDET_CKPT")
    if env_ckpt:
        env_path = Path(env_ckpt)
        if env_path.is_file() and env_path.stat().st_size > 0:
            print(f"Using checkpoint from RTMDET_CKPT: {env_path}")
            return env_path

    cached = Path("/work/.modelcache") / CHECKPOINT_PATH.name
    if cached.is_file() and cached.stat().st_size > 0:
        print(f"Using cached checkpoint: {cached}")
        return cached

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    if not CHECKPOINT_PATH.exists():
        run(["curl", "-L", "-o", str(CHECKPOINT_PATH), CHECKPOINT_URL])
    return CHECKPOINT_PATH


def export_coreml(deploy_cfg: Path, checkpoint_path: Path, image_path: Path) -> Path:
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        str(MMDEPLOY_DIR / "tools/deploy.py"),
        str(deploy_cfg),
        str(MODEL_CONFIG),
        str(checkpoint_path),
        str(image_path),
        "--work-dir",
        str(WORK_DIR),
        "--device",
        "cpu",
        "--log-level",
        "INFO",
    ]
    print("+", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, check=False)

    packages = sorted(WORK_DIR.glob("*.mlpackage"))
    if not packages:
        if result.returncode:
            raise RuntimeError(f"MMDeploy failed with exit code {result.returncode}")
        raise RuntimeError(f"No .mlpackage produced in {WORK_DIR}")
    if result.returncode:
        print(
            f"MMDeploy exited with {result.returncode} after producing {packages[0]}; "
            "continuing because Linux cannot run Core ML prediction for visualization.",
            flush=True,
        )
    return packages[0]


def install_into_ios(model_package: Path) -> Path:
    IOS_DROP_DIR.mkdir(parents=True, exist_ok=True)
    destination = IOS_DROP_DIR / "rtmdet-ins-m.mlpackage"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(model_package, destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Export RTMDet-Ins-m to Core ML.")
    parser.add_argument("--image", type=Path, default=TEST_IMAGE, help="Test image for conversion tracing")
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Missing test image: {args.image}")
    if not MODEL_CONFIG.exists():
        raise FileNotFoundError(f"Missing mmdetection config: {MODEL_CONFIG}")
    if not MMDEPLOY_DIR.exists():
        raise FileNotFoundError(f"Missing mmdeploy checkout: {MMDEPLOY_DIR}")

    deploy_cfg = ensure_deploy_config()
    checkpoint = ensure_checkpoint()
    model_package = export_coreml(deploy_cfg, checkpoint, args.image)
    installed = install_into_ios(model_package)
    print()
    print("Export complete")
    print(f"Core ML package: {model_package}")
    print(f"Installed for iOS: {installed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
