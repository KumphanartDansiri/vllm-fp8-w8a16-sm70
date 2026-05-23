# Self-contained env for the V100 FP8 -> FP16 dequant test.
# CUDA 12.4 to match host driver (535.288.01) and toolkit at /usr/local/cuda-12.4.
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev ninja-build git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CUDA 12.4 wheels. Tested versions known to ship torch.float8_e4m3fn.
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.5.1

WORKDIR /work
