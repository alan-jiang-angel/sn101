FROM python:3.12-slim

ARG IMAGE_REVISION=unknown

LABEL org.opencontainers.image.title="tag101"
LABEL org.opencontainers.image.revision="${IMAGE_REVISION}"

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV IMAGE_REVISION="${IMAGE_REVISION}"

WORKDIR /app

# fasttext builds a native extension and needs a C++17-capable toolchain.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt . \
    && mkdir -p /home/node/.bittensor/wallets /home/node/state

CMD ["tag101-validator", "--help"]
