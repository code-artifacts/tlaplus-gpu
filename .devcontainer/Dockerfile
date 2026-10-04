FROM mcr.microsoft.com/devcontainers/base:ubuntu

ENV DEBIAN_FRONTEND=noninteractive

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        unzip \
        zip \
        tar \
        git \
        openjdk-17-jdk \
        maven \
        ant \
        xvfb; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*



