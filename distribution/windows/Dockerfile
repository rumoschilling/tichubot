FROM debian:bookworm-slim

ARG PYTHON_VERSION=3.12.10
ARG TORCH_VERSION=2.11.0
ARG NUMPY_VERSION=2.4.3
ARG PYINSTALLER_VERSION=6.21.0

ENV DEBIAN_FRONTEND=noninteractive \
    WINEARCH=win64 \
    WINEDEBUG=-all \
    WINEPREFIX=/opt/wineprefix

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        wine \
        xauth \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

ARG EMBED_PYTHON_VERSION=3.11.9

RUN apt-get update \
    && apt-get install -y --no-install-recommends unzip \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p "$WINEPREFIX/drive_c/Python311" \
    && wget -q "https://www.python.org/ftp/python/${EMBED_PYTHON_VERSION}/python-${EMBED_PYTHON_VERSION}-embed-amd64.zip" -O /tmp/python.zip \
    && unzip -q /tmp/python.zip -d "$WINEPREFIX/drive_c/Python311" \
    && sed -i 's/^#import site/import site/' "$WINEPREFIX/drive_c/Python311/python311._pth" \
    && wget -q "https://bootstrap.pypa.io/get-pip.py" -O /tmp/get-pip.py \
    && xvfb-run -a wine 'C:\Python311\python.exe' 'Z:\tmp\get-pip.py' \
    && rm /tmp/python.zip /tmp/get-pip.py

RUN xvfb-run -a wine C:\\Python311\\python.exe -m pip install --no-cache-dir \
        "numpy==${NUMPY_VERSION}" \
        "pyinstaller==${PYINSTALLER_VERSION}"

RUN xvfb-run -a wine C:\\Python311\\python.exe -m pip install --no-cache-dir \
        "filelock==3.29.0" \
        "fsspec==2026.4.0" \
        "jinja2==3.1.6" \
        "MarkupSafe==3.0.3" \
        "mpmath==1.3.0" \
        "networkx==3.6.1" \
        "setuptools==78.1.0" \
        "sympy==1.14.0" \
        "typing-extensions==4.15.0"

COPY .cache/torch-2.11.0+cpu-cp311-cp311-win_amd64.whl /tmp/torch-2.11.0+cpu-cp311-cp311-win_amd64.whl
RUN xvfb-run -a wine C:\\Python311\\python.exe -m pip install --no-cache-dir --no-deps \
        'Z:\tmp\torch-2.11.0+cpu-cp311-cp311-win_amd64.whl' \
    && rm /tmp/torch-2.11.0+cpu-cp311-cp311-win_amd64.whl

# NumPy 2.4 uses ucrt complex-number entry points that Wine 8 does not yet
# implement. The model uses only the stable NumPy array API during inference.
COPY .cache/numpy-1.26.4-cp311-cp311-win_amd64.whl /tmp/numpy-1.26.4-cp311-cp311-win_amd64.whl
RUN xvfb-run -a wine C:\\Python311\\python.exe -m pip install --no-cache-dir --force-reinstall \
        'Z:\tmp\numpy-1.26.4-cp311-cp311-win_amd64.whl' \
    && rm /tmp/numpy-1.26.4-cp311-cp311-win_amd64.whl

COPY entrypoint.sh /usr/local/bin/tichubot-windows-build
RUN chmod +x /usr/local/bin/tichubot-windows-build

ENTRYPOINT ["/usr/local/bin/tichubot-windows-build"]
