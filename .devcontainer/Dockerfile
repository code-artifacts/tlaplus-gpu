FROM mcr.microsoft.com/devcontainers/base:ubuntu

## Suggested by ChatGPT
RUN set -eux; \
    sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list; \
    sed -i 's|http://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list; \
    apt-get update; \
    apt-get update; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get -y install --no-install-recommends \
      -o Acquire::Retries=5 \
      -o Acquire::http::Timeout="60" \
      -o Acquire::https::Timeout="60" \
      -o Acquire::http::Pipeline-Depth=0 \
      --fix-missing \
      ca-certificates curl unzip zip tar \
      git \
      openjdk-11-jdk \
      openjdk-17-jdk \
      maven \
      ant \
      xvfb; \
    apt-get clean -y; \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --set java /usr/lib/jvm/java-17-openjdk-amd64/bin/java && \
    update-alternatives --set javac /usr/lib/jvm/java-17-openjdk-amd64/bin/javac

# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends ant xvfb \
#     && apt-get clean -y && rm -rf /var/lib/apt/lists/*