# Set the base image
FROM debian:bookworm-slim AS builder

# Install system dependencies. The PyPI `ta-lib` wheel bundles the TA-Lib C library for Linux
# (verified: it imports and computes correctly with no system libta-lib installed), unlike conda's
# `libta-lib` conda-forge package which this used to need separately. gcc/g++/make/python3-dev are
# a source-build safety net for whichever dependency doesn't ship a wheel for this interpreter/arch.
RUN apt-get update && \
    apt-get install -y sudo libusb-1.0-0 gcc g++ python3-dev make curl && \
    rm -rf /var/lib/apt/lists/*

# Install pixi (from official installer)
RUN curl -fsSL https://pixi.sh/install.sh | bash && \
    cp /root/.pixi/bin/pixi /usr/local/bin/pixi

WORKDIR /home/hummingbot

# Copy project files
COPY . /home/hummingbot

# Install dependencies via pixi
RUN pixi install && pixi run install-dev

# Build Cython extensions
RUN pixi run python setup.py build_ext --inplace -j 8 && \
    rm -rf build/ && \
    find . -type f -name "*.cpp" -delete


# Build final image using artifacts from builder
FROM debian:bookworm-slim AS release

# Dockerfile author / maintainer
LABEL maintainer="Fede Cardoso @dardonacci <federico@hummingbot.org>"

# Build arguments
ARG BRANCH=""
ARG COMMIT=""
ARG BUILD_DATE=""
LABEL branch=${BRANCH}
LABEL commit=${COMMIT}
LABEL date=${BUILD_DATE}

# Set ENV variables
ENV COMMIT_SHA=${COMMIT}
ENV COMMIT_BRANCH=${BRANCH}
ENV BUILD_DATE=${BUILD_DATE}

ENV INSTALLATION_TYPE=docker

# Install system dependencies (the TA-Lib C library ships bundled inside the `ta-lib` wheel).
# libstdc++6 is what the compiled Cython/C++ extensions link against; the base image already has it
# (apt depends on it), but it is listed so the image does not rely on that.
RUN apt-get update && \
    apt-get install -y sudo libusb-1.0-0 libstdc++6 && \
    rm -rf /var/lib/apt/lists/*

# Create mount points
RUN mkdir -p /home/hummingbot/conf /home/hummingbot/conf/connectors /home/hummingbot/conf/strategies /home/hummingbot/conf/controllers /home/hummingbot/conf/scripts /home/hummingbot/logs /home/hummingbot/data /home/hummingbot/certs /home/hummingbot/scripts /home/hummingbot/controllers

WORKDIR /home/hummingbot

# Copy all build artifacts from builder image
COPY --from=builder /root/.pixi /root/.pixi
COPY --from=builder /usr/local/bin/pixi /usr/local/bin/pixi
COPY --from=builder /home/hummingbot /home/hummingbot

# Set the default command to run when starting the container
CMD pixi run python ./bin/hummingbot_quickstart.py 2>> ./logs/errors.log
