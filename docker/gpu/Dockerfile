FROM nvidia/cuda:11.2.0-cudnn8-devel AS compile

# Install dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y \
    git \
    cmake \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-numpy \
    gcc \
    build-essential \
    gfortran \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*


# Virtual Environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Scikit learn
RUN pip3 install --upgrade pip && \
    pip3 install scikit-build

# Install dlib
ENV CFLAGS=-static
RUN git clone -b 'v19.21' --single-branch https://github.com/davisking/dlib.git dlib/ && \
    mkdir -p /dlib/build && \
    cmake -H/dlib -B/dlib/build -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1 && \
    cmake --build /dlib/build && \
    cd /dlib && \
    python3 /dlib/setup.py install --set BUILD_SHARED_LIBS=OFF

# Install face recognition
RUN pip3 install face_recognition

# Runtime Image
FROM nvidia/cuda:11.2.0-cudnn8-runtime

# Install requirements
RUN apt-get update && apt-get install -y \
    python3 \
    python3-distutils

# Copy in libs
COPY --from=compile /opt/venv /opt/venv
COPY --from=compile \
    # Sources
    /lib/x86_64-linux-gnu/libpthread.so.0 \
    /lib/x86_64-linux-gnu/libdl.so.2 \
    /lib/x86_64-linux-gnu/librt.so.1 \
    /lib/x86_64-linux-gnu/libX11.so.6 \
    /lib/x86_64-linux-gnu/libpng16.so.16 \
    /lib/x86_64-linux-gnu/libjpeg.so.8 \
    /lib/x86_64-linux-gnu/libcudnn.so.8 \
    /lib/x86_64-linux-gnu/libstdc++.so.6 \
    /lib/x86_64-linux-gnu/libm.so.6 \
    /lib/x86_64-linux-gnu/libgcc_s.so.1 \
    /lib/x86_64-linux-gnu/libc.so.6 \
    /lib/x86_64-linux-gnu/libxcb.so.1 \
    /lib/x86_64-linux-gnu/libz.so.1 \
    /lib/x86_64-linux-gnu/libXau.so.6 \
    /lib/x86_64-linux-gnu/libXdmcp.so.6 \
    /lib/x86_64-linux-gnu/libbsd.so.0 \
    # Destination
    /lib/x86_64-linux-gnu/
COPY --from=compile \
    # Sources
    /usr/local/cuda/lib64/libcublas.so.11 \
    /usr/local/cuda/lib64/libcurand.so.10 \
    /usr/local/cuda/lib64/libcusolver.so.11 \
    /usr/local/cuda/lib64/libcublasLt.so.11 \
    # Destination
    /usr/local/cuda/lib64/

# Add our packages
ENV PATH="/opt/venv/bin:$PATH"
