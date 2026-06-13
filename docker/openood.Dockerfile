# CPU environment for the strict-blind OpenOOD oracle.
#
# Build:
#   docker build -f docker/openood.Dockerfile -t repro-openood:latest .
#
# The clean OpenOOD repository, official data, and checkpoints are mounted at
# runtime. The canonical CPU runner and private result are never copied in.

FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash git ripgrep libgl1 libglib2.0-0 libxext6 libsm6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        torch torchvision \
        numpy==1.26.4 scipy scikit-learn pandas \
        pyyaml tqdm pillow matplotlib \
        opencv-python-headless imgaug diffdist gdown

RUN python -c "import torch, torchvision, numpy, sklearn, pandas, cv2, imgaug; \
    print('repro-openood OK | torch', torch.__version__, '| numpy', numpy.__version__)"
