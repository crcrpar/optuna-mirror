ARG PYTHON_VERSION=3.7

FROM python:${PYTHON_VERSION}

RUN apt-get update \
    && apt-get -y install openmpi-bin libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -U 'pip<20' \
    && pip install --no-cache-dir --progress-bar off -U setuptools

WORKDIR /workspaces
COPY . .

ARG BUILD_TYPE=''

RUN if [ "${BUILD_TYPE}" = "dev" ]; then \
        pip install --no-cache-dir -e '.[checking, doctest, document, example, testing]'; \
    else \
        pip install --no-cache-dir -e .; \
    fi \
    && pip install jupyter notebook