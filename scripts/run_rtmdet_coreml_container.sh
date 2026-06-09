#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed" >&2
  exit 127
fi

if ! docker version >/dev/null 2>&1; then
  echo "docker daemon is not running. Start Colima first:" >&2
  echo "  colima start --arch x86_64 --cpu 4 --memory 8" >&2
  exit 1
fi

docker run --rm -i --platform linux/amd64 \
  -v "$REPO_ROOT:/work" \
  -v "$REPO_ROOT/.wheelcache:/work/.wheelcache" \
  -v "$REPO_ROOT/.modelcache:/work/.modelcache" \
  -w /work \
  python:3.10-slim \
  bash -se <<'CONTAINER_SCRIPT'
    set -eu
    export DEBIAN_FRONTEND=noninteractive
    export PIP_NO_CACHE_DIR=1
    export PYTHONUNBUFFERED=1
    WHEELCACHE=/work/.wheelcache
    MODELCACHE=/work/.modelcache
    mkdir -p "$WHEELCACHE" "$MODELCACHE"
    export PIP_CACHE_DIR="$WHEELCACHE/pipcache"
    export PIP_DEFAULT_TIMEOUT=60
    export PIP_RETRIES=5

    # Preflight: show cache hits before paying install cost.
    _stat () {
      local label="$1" glob="$2" f
      f=$(ls -S $glob 2>/dev/null | head -n1 || true)
      if [ -n "$f" ] && [ -s "$f" ]; then
        printf "  HOT    %-14s %s (%s)\n" "$label" "$(basename "$f")" "$(du -h "$f" | cut -f1)"
      else
        printf "  FETCH  %-14s (will download/build)\n" "$label"
      fi
    }

    echo "---------------- PREFLIGHT ----------------"
    echo "Cache artifacts:"
    _stat "torch"        "$WHEELCACHE/torch-2.1.2*.whl"
    _stat "torchvision"  "$WHEELCACHE/torchvision-0.16.2*.whl"
    _stat "coremltools"  "$WHEELCACHE/coremltools-7.1*.whl"
    _stat "mmcv"         "$WHEELCACHE/mmcv-2.1.0-cp310-*.whl"
    _stat "checkpoint"   "$MODELCACHE/rtmdet-ins_m_*.pth"
    echo "-------------------------------------------"

    apt-get update
    apt-get install -y --no-install-recommends \
      git \
      curl \
      ca-certificates \
      build-essential \
      cmake \
      gcc-12 \
      g++-12 \
      libgl1 \
      libglib2.0-0
    rm -rf /var/lib/apt/lists/*

    _net () {
      if curl -s -I --connect-timeout 6 --max-time 10 "$2" >/dev/null 2>&1; then
        printf "  UP    %s\n" "$1"
      else
        printf "  DOWN  %s\n" "$1"
      fi
    }

    echo "------------- REACHABILITY ---------------"
    _net "GitHub      (repos)"      "https://github.com"
    _net "PyPI        (sdist)"      "https://pypi.org/simple"
    _net "pytorch.org (cpu wheels)" "https://download.pytorch.org/whl/cpu"
    _net "OpenMMLab   (mmcv/ckpt)"  "https://download.openmmlab.com"
    echo "-------------------------------------------"

    mkdir -p /opt/openmmlab
    cd /opt/openmmlab
    git clone --depth 1 --branch v3.3.0 https://github.com/open-mmlab/mmdetection.git

    python -m pip install --upgrade pip "setuptools<70" wheel ninja

    echo "-------------- TOOLCHAIN -----------------"
    printf "  g++-12: %s | ninja: %s | nproc: %s\n" \
      "$(command -v g++-12 || echo MISSING)" \
      "$(command -v ninja || echo MISSING)" \
      "$(nproc)"
    echo "-------------------------------------------"

    cache_pip() {
      spec="$1"
      shift || true
      name="${spec%%[<>=]*}"
      import_name=$(printf "%s" "$name" | tr "-" "_")
      if python -c "import ${import_name}" 2>/dev/null; then
        echo "$name already importable"
        return 0
      fi
      if ! ls "$WHEELCACHE"/${name}-*.whl >/dev/null 2>&1; then
        echo "Caching wheel for $spec"
        python -m pip wheel "$spec" -w "$WHEELCACHE" "$@" || {
          echo "Could not build/fetch wheel for $spec"
          return 1
        }
      fi
      python -m pip install --no-deps "$WHEELCACHE"/${name}-*.whl
    }

    TORCH_IDX=https://download.pytorch.org/whl/cpu
    cache_pip "torch==2.1.2" --index-url "$TORCH_IDX"
    cache_pip "torchvision==0.16.2" --index-url "$TORCH_IDX"

    # Hard numpy pin and deterministic mmcv source build fallback.
    set -euo pipefail
    CACHE="/work/.wheelcache"
    mkdir -p "$CACHE"

    echo ">>> pinning numpy==1.26.4 (before any mmcv/mmdet import)"
    pip install --force-reinstall "numpy==1.26.4"
    python -c "import numpy; assert numpy.__version__.startswith('1.26'), numpy.__version__; print('numpy', numpy.__version__, 'OK')"

    # Resolve dependencies normally, preferring host-mounted cached wheels.
    pip install --find-links "$CACHE" "torch==2.1.2+cpu" "torchvision==0.16.2+cpu"
    pip install --find-links "$CACHE" "coremltools==7.1" "mmengine==0.10.3" "mmdet==3.3.0"
    pip install --force-reinstall --no-deps "numpy==1.26.4"

    MMCV_VER=2.1.0; PYTAG=cp310
    WHEEL="mmcv-${MMCV_VER}-${PYTAG}-${PYTAG}-manylinux1_x86_64.whl"
    WHEEL_URL="https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/${WHEEL}"
    COMPILED_MMCV=$(ls "$CACHE"/mmcv-${MMCV_VER}-${PYTAG}-*.whl 2>/dev/null | head -n1 || true)

    if python -c "import mmcv, mmcv.ops" 2>/dev/null; then
      echo ">>> mmcv ops already present"
    elif [ -n "$COMPILED_MMCV" ] && [ -s "$COMPILED_MMCV" ]; then
      echo ">>> installing cached mmcv wheel"
      pip install --no-deps "$COMPILED_MMCV"
    else
      rm -f "$CACHE"/mmcv-${MMCV_VER}-py2.py3-none-any.whl
      if curl -L --fail --retry 2 --retry-all-errors \
           --connect-timeout 10 --max-time 60 -C - \
           -o "$CACHE/$WHEEL" "$WHEEL_URL"; then
        echo ">>> got prebuilt mmcv wheel from OpenMMLab"
        pip install --no-deps "$CACHE/$WHEEL"
      else
        echo ">>> OpenMMLab unreachable; building mmcv from PyPI sdist"
        export MMCV_WITH_OPS=1 FORCE_CUDA=0 MAX_JOBS="${MMCV_MAX_JOBS:-1}"
        echo ">>> mmcv MAX_JOBS=$MAX_JOBS"
        command -v g++-12 >/dev/null 2>&1 && export CC=gcc-12 CXX=g++-12
        pip install -U "setuptools<70" wheel ninja
        pip wheel --no-cache-dir --no-build-isolation --no-deps "mmcv==${MMCV_VER}" -w "$CACHE"
        COMPILED_MMCV=$(ls "$CACHE"/mmcv-${MMCV_VER}-${PYTAG}-*.whl 2>/dev/null | head -n1 || true)
        if [ -z "$COMPILED_MMCV" ]; then
          echo "mmcv source build did not produce a compiled cp310 wheel"
          ls -lh "$CACHE"/mmcv-${MMCV_VER}-*.whl 2>/dev/null || true
          exit 1
        fi
        pip install --no-deps "$COMPILED_MMCV"
      fi
    fi

    pip install --force-reinstall --no-deps "numpy==1.26.4"

    python - <<'PY'
import numpy, mmcv
from mmcv.ops import nms
assert numpy.__version__.startswith("1.26"), numpy.__version__
print("numpy", numpy.__version__, "| mmcv", mmcv.__version__, "- ops OK")
PY

    CKPT=rtmdet-ins_m_8xb32-300e_coco_20221123_001039-6eba602e.pth
    CKPT_URL=https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_m_8xb32-300e_coco/${CKPT}
    if [ ! -s "$MODELCACHE/$CKPT" ]; then
      echo "Fetching checkpoint"
      curl -L --fail --retry 5 --retry-all-errors \
        --connect-timeout 15 --max-time 600 -C - \
        -o "$MODELCACHE/$CKPT" "$CKPT_URL" || {
          echo "Checkpoint host unreachable — place it manually at $MODELCACHE/$CKPT"
          exit 1
        }
    fi
    export RTMDET_CKPT="$MODELCACHE/$CKPT"

    cd /work
    python scripts/rtmdet_ins_coreml_raw_export.py
CONTAINER_SCRIPT
