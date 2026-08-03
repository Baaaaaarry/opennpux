ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    bison \
    build-essential \
    ca-certificates \
    clang \
    cmake \
    curl \
    default-jdk-headless \
    flex \
    g++ \
    gcc \
    git \
    libfl-dev \
    libz-dev \
    lld \
    ninja-build \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv \
    srecord \
    unzip \
    verilator \
    wget \
    xz-utils \
    zip \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /usr/local/bin/bazel \
    https://github.com/bazelbuild/bazelisk/releases/download/v1.22.0/bazelisk-linux-amd64 \
 && chmod +x /usr/local/bin/bazel

WORKDIR /workspace
