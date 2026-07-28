ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git ffmpeg curl build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/matrixgame-systems
COPY . .
RUN python -m pip install --upgrade pip && \
    python -m pip install -e '.[gpu,service]'

EXPOSE 8000
ENTRYPOINT ["mgs-serve"]
CMD ["--config", "configs/service.yaml"]
