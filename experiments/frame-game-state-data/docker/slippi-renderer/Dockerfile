FROM ubuntu:22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG ISHIIRUKA_REF=slippi
ARG ISHIIRUKA_REPO=https://github.com/project-slippi/Ishiiruka.git
ARG BUILD_JOBS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    cmake \
    curl \
    file \
    g++ \
    gcc \
    git \
    make \
    patch \
    pkg-config \
    python3 \
    wget \
    libao-dev \
    libasound2-dev \
    libavcodec-dev \
    libavformat-dev \
    libbluetooth-dev \
    libenet-dev \
    libgtk2.0-dev \
    liblzo2-dev \
    libminiupnpc-dev \
    libopenal-dev \
    libpulse-dev \
    libreadline-dev \
    libsfml-dev \
    libsoil-dev \
    libsoundtouch-dev \
    libswscale-dev \
    libusb-1.0-0-dev \
    libwebkit2gtk-4.0-dev \
    libxext-dev \
    libxrandr-dev \
    portaudio19-dev \
    zlib1g-dev \
    libudev-dev \
    libevdev-dev \
    libmbedtls-dev \
    libcurl4-openssl-dev \
    libegl1-mesa-dev \
    libpng-dev \
    qtbase5-private-dev \
    libxxf86vm-dev \
    x11proto-xinerama-dev \
    libfuse2 \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain stable

ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /opt/slippi-src
RUN git clone --branch "${ISHIIRUKA_REF}" --depth 1 "${ISHIIRUKA_REPO}" Ishiiruka
WORKDIR /opt/slippi-src/Ishiiruka
RUN git submodule update --init --recursive

COPY patches/ishiiruka-macos-png-frame-dump.patch /tmp/ishiiruka-png-frame-dump.patch
RUN patch -p1 < /tmp/ishiiruka-png-frame-dump.patch

RUN mkdir -p build && cd build && \
    cmake \
      -DLINUX_LOCAL_DEV=true \
      -DIS_PLAYBACK=true \
      -DENABLE_ANALYTICS=false \
      -DCMAKE_BUILD_TYPE=Release \
      .. && \
    make -j"${BUILD_JOBS}"

RUN rm -rf build/Binaries/Sys && \
    cp -r Data/Sys build/Binaries/Sys && \
    rm -rf build/Binaries/Sys/GameSettings && \
    mkdir -p build/Binaries/Sys/GameSettings && \
    cp -r Data/PlaybackGeckoCodes/. build/Binaries/Sys/GameSettings/ && \
    grep -q '\$Required: Slippi Playback' build/Binaries/Sys/GameSettings/GALE01r2.ini && \
    touch build/Binaries/portable.txt

FROM ubuntu:22.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libao4 \
    libasound2 \
    libavcodec58 \
    libavformat58 \
    libbluetooth3 \
    libcurl4 \
    libegl1-mesa \
    libevdev2 \
    libgtk-3-0 \
    libgtk2.0-0 \
    libopengl0 \
    liblzo2-2 \
    libmbedtls14 \
    libminiupnpc17 \
    libopenal1 \
    libpng16-16 \
    libpulse0 \
    libreadline8 \
    libsfml-network2.5 \
    libsfml-system2.5 \
    libsoil1 \
    libsoundtouch1 \
    libswscale5 \
    libusb-1.0-0 \
    libwebkit2gtk-4.0-37 \
    libxext6 \
    libxi6 \
    libxrandr2 \
    libxxf86vm1 \
    mesa-utils \
    openssh-server \
    python3 \
    libportaudio2 \
    rsync \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/slippi
COPY --from=builder /opt/slippi-src/Ishiiruka/build/Binaries/ /opt/slippi/
COPY --from=builder /opt/slippi-src/Ishiiruka/build/Source/Core/DolphinWX/libslippi_rust_extensions.so /opt/slippi/
COPY docker/scripts/render-replay.sh /opt/slippi-renderer/render-replay.sh
COPY docker/scripts/start-runpod-worker.sh /opt/slippi-renderer/start-runpod-worker.sh
RUN chmod +x /opt/slippi-renderer/render-replay.sh /opt/slippi-renderer/start-runpod-worker.sh && \
    test -x /opt/slippi/dolphin-emu && \
    ln -sf /opt/slippi/dolphin-emu "/opt/slippi/Slippi Dolphin" && \
    grep -q '\$Required: Slippi Playback' /opt/slippi/Sys/GameSettings/GALE01r2.ini

ENV SLIPPI_DOLPHIN_BIN=/opt/slippi/dolphin-emu
ENV LD_LIBRARY_PATH=/opt/slippi
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV MESA_GL_VERSION_OVERRIDE=3.3

ENTRYPOINT ["/opt/slippi-renderer/render-replay.sh"]
