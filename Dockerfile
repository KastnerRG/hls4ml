FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MINICONDA=Miniconda3-py310_25.1.1-2-Linux-x86_64.sh

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash build-essential ca-certificates curl file git iproute2 locales \
    libasound2 libdrm2 libegl1 libfontconfig1 libfreetype6 libgbm1 libglib2.0-0 \
    libgl1 libglu1-mesa libgtk-3-0 libice6 libnss3 libpixman-1-0 libpng16-16 \
    libsm6 libstdc++6 libtinfo5 libtinfo6 libuuid1 libx11-6 libxcomposite1 \
    libxcursor1 libxdamage1 libxext6 libxfixes3 libxi6 libxinerama1 libxkbcommon0 \
    libxrandr2 libxrender1 libxtst6 make procps unzip xauth xterm zip zlib1g \
    && rm -rf /var/lib/apt/lists/*

RUN sed -i 's/^# \(en_US.UTF-8 UTF-8\)/\1/' /etc/locale.gen && locale-gen

RUN curl -fsSLo /tmp/miniconda.sh "https://repo.anaconda.com/miniconda/${MINICONDA}" \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda clean -afy

ENV CONDA_DIR=/opt/conda \
    HLS4ML_CONDA_ENV=/opt/conda/envs/hls4ml-vek280 \
    HLS4ML_VITIS_ROOT=/tools/Xilinx/Vivado/2025.2 \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    PATH=/opt/conda/bin:${PATH}

COPY environment-vek280.yml /tmp/environment-vek280.yml
RUN sed '/- -e \.$/d' /tmp/environment-vek280.yml > /tmp/environment-docker.yml \
    && conda env create -f /tmp/environment-docker.yml \
    && conda clean -afy \
    && rm /tmp/environment-*.yml

RUN mkdir -p /workspace /home/hls4ml \
    && printf '%s\n' \
        'if [ -f "${HLS4ML_VITIS_ROOT}/Vitis/settings64.sh" ]; then' \
        '    source "${HLS4ML_VITIS_ROOT}/Vitis/settings64.sh" >/dev/null 2>&1' \
        'fi' > /home/hls4ml/.bash_profile

COPY --chmod=755 docker/enter.sh /usr/local/bin/hls4ml-shell
COPY docker/bashrc /etc/hls4ml.bashrc

WORKDIR /workspace
CMD ["sleep", "infinity"]
