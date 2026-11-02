# ------------------------------------------------------------
# Base image: CUDA 11.8 (compatible with Detectron2 prebuilt wheels)
# ------------------------------------------------------------
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    bzip2 \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    ninja-build \
    python3 \
    python3-dev \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Make python3 the default
RUN ln -s /usr/bin/python3 /usr/bin/python

# ------------------------------------------------------------
# Miniconda setup
# ------------------------------------------------------------
RUN curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /opt/conda \
    && rm -rf /tmp/miniconda.sh

ENV PATH="/opt/conda/bin:${PATH}"

# Accept TOS
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Update conda
RUN conda update -n base -c defaults conda -y && conda clean --all --yes

# Create leap env
COPY ./environment.yml /leap/environment.yml
RUN conda env create -f /leap/environment.yml && conda clean --all --yes

# Create retinanet env
COPY ./environment-retina.yml /leap/environment-retina.yml
RUN conda env create -f /leap/environment-retina.yml && conda clean --all --yes

# Path to the leap environment Python/pip
ENV LEAP_PYTHON=/opt/conda/envs/leap/bin/python
ENV LEAP_PIP=/opt/conda/envs/leap/bin/pip
ENV PATH="/opt/conda/envs/leap/bin:${PATH}"

# ------------------------------------------------------------
# Core Python dependencies
# ------------------------------------------------------------

# Upgrade pip, setuptools, wheel inside leap env
RUN $LEAP_PIP install --upgrade pip setuptools wheel

# ------------------------------------------------------------
# Build Detectron2 from source
# ------------------------------------------------------------
# Pinned to the commit the Tetris runs' leap env had (env/leap.pip.txt).
RUN git clone https://github.com/facebookresearch/detectron2.git /leap/detectron2_repo \
    && git -C /leap/detectron2_repo checkout 8a9d885b3d4dcf1bef015f0593b872ed8d32b4ab
WORKDIR /leap/detectron2_repo

# Install Detectron2 in-place, using leap env and no build isolation
RUN $LEAP_PYTHON -m pip install --no-cache-dir --no-build-isolation -e .

# ------------------------------------------------------------
# Install remaining Python requirements
# ------------------------------------------------------------
WORKDIR /leap
COPY ./requirements.txt /leap/requirements.txt
RUN $LEAP_PIP install --no-cache-dir --prefer-binary -r /leap/requirements.txt

# Installed by hand into the container of the Tetris runs (env/leap.conda.yml).
RUN conda install -y -n leap -c pytorch faiss-gpu=1.7.4 && conda clean --all --yes
RUN $LEAP_PIP install --no-cache-dir yacs==0.1.8 fvcore==0.1.5.post20221221

# ------------------------------------------------------------
# Copy LEAP code
# ------------------------------------------------------------
COPY . .

# Install LEAP package in editable mode
RUN $LEAP_PIP install -e .

# ------------------------------------------------------------
# Environment variables
# ------------------------------------------------------------
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# ------------------------------------------------------------
# Sanity check
# ------------------------------------------------------------
RUN $LEAP_PYTHON - << 'EOF'
import torch
import detectron2
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("WARNING: CUDA not available during build. This is normal if building without nvidia-runtime.")
print("Torch version:", torch.__version__)
print("Detectron2 version:", detectron2.__version__)
EOF

# ------------------------------------------------------------
# Default command
# ------------------------------------------------------------
CMD ["/bin/bash"]
