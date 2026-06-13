# Reproducible environment for the mmpretrain clone-and-navigate oracle.
#
# This image is the *pre-provisioned env* the agent runs INSIDE — it is NOT built
# by the agent. The agent's job is navigation + running the repo's eval; building
# mmcv is the documented, irreducible env-block (mmcv won't build on Py3.12 and
# has no arm64 prebuilt wheel), so we solve it here, once, with PREBUILT x86_64
# wheels — no source build.
#
# Build (on Apple Silicon / arm64 the x86_64 wheels need amd64 emulation):
#   docker build --platform linux/amd64 -f docker/mmpretrain.Dockerfile -t repro-mmpretrain:latest .
#
# The manifest evals/benchmark/mmpretrain_resnet18_cifar10.yaml references this
# image by name (docker_image: repro-mmpretrain:latest).

FROM --platform=linux/amd64 python:3.10-slim

# opencv-python (pulled in by mmcv) needs these shared libs at import time,
# otherwise: "libxcb.so.1: cannot open shared object file".
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libxext6 libsm6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# torch 2.1.0 (CPU) — the exact version mmcv 2.1.0 ships a prebuilt wheel for.
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 \
        --index-url https://download.pytorch.org/whl/cpu

# mmcv via its PREBUILT wheel (openmim resolves the torch-matched build), then
# mmengine + mmpretrain. This is the whole point: no compiling mmcv from source.
RUN pip install --no-cache-dir -U openmim \
    && mim install mmengine "mmcv==2.1.0" \
    && mim install mmpretrain

# Pin numpy<2 LAST: mmpretrain's deps re-upgrade numpy to 2.x, which breaks the
# torch-2.1.0 (built against numpy 1) ABI at the first tensor<->numpy conversion.
RUN pip install --no-cache-dir numpy==1.26.4

# Build-time smoke check: a broken env fails `docker build`, not a later agent run.
RUN python -c "import torch, numpy, mmcv, mmengine, mmpretrain; \
    a = torch.randn(2, 3).numpy(); assert a.shape == (2, 3); \
    print('repro-mmpretrain OK | mmpretrain', mmpretrain.__version__, \
          '| torch', torch.__version__, '| mmcv', mmcv.__version__, \
          '| numpy', numpy.__version__)"
