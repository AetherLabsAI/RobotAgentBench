FROM node@sha256:3ed98443660ff48413b5af19ecefce3b9d50845a901a43420d47cc63a9fddbe9 AS node
FROM python@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3 AS python
FROM ubuntu@sha256:68ee22004a9109b44973313b02ff7b1de250f1de0f5ecb628d73f579aaff33ac
COPY --from=python /usr/local /usr/local
COPY --from=node /usr/local/bin/node /usr/local/bin/node
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libffi8 libssl3t64 zlib1g libbz2-1.0 libreadline8t64 \
    libsqlite3-0 liblzma5 libexpat1 libuuid1 libgdbm6t64 libncursesw6 \
    libstdc++6 libgomp1 libglib2.0-0t64 libx11-6 libxext6 libxrender1 libxi6 \
    libxrandr2 libxtst6 libxt6t64 libxcursor1 libxinerama1 libxkbcommon0 libsm6 libice6 \
    libgl1 libegl1 libopengl0 libglu1-mesa libvulkan1 libnss3 libasound2t64 \
    libfontconfig1 libfreetype6 libcurl4t64 && ldconfig
COPY docker/requirements-native.txt /opt/robotagentbench/requirements-native.txt
# The canonical SDK path also supplies an original resource namespace inside
# the image. /opt/robotagentbench/native is the user-facing installation path.
RUN /usr/local/bin/python3.11 -m pip install --no-cache-dir \
    --target /workspace/lukun/arena_software/isaac51-py311/lib/python3.11/site-packages \
    --extra-index-url https://pypi.nvidia.com -r /opt/robotagentbench/requirements-native.txt \
    && ln -s /workspace/lukun/arena_software/isaac51-py311/lib/python3.11/site-packages /opt/robotagentbench/native
COPY pyproject.toml README.md LICENSE /opt/robotagentbench/package/
COPY src /opt/robotagentbench/package/src
RUN /usr/local/bin/python3.11 -m pip install --no-cache-dir --no-deps /opt/robotagentbench/package
# Seal the original SDK core hashes outside the two per-episode writable
# overlays. This CPU-only build step imports no simulator or device library.
RUN /usr/local/bin/python3.11 -B -m robotagentbench.simulation.native_deployment image-baseline \
    --sdk-site /opt/robotagentbench/native \
    --destination /opt/robotagentbench/native-baseline.json
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
WORKDIR /tmp
