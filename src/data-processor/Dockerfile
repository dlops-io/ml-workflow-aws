# Use the official Debian-hosted Python image
FROM python:3.11-slim

ARG DEBIAN_PACKAGES="build-essential git curl wget unzip gzip"

ENV DEBIAN_FRONTEND=noninteractive

ENV LANG=C.UTF-8

ENV PYENV_SHELL=/bin/bash

ENV PYTHONUNBUFFERED=1

RUN set -ex; \
    for i in $(seq 1 8); do mkdir -p "/usr/share/man/man${i}"; done && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends $DEBIAN_PACKAGES && \
    apt-get install -y --no-install-recommends apt-transport-https ca-certificates gnupg curl && \
    pip install --no-cache-dir --upgrade pip && \
    pip install pipenv  && \
    mkdir -p /app && \
    mkdir -p /persistent

WORKDIR /app

ADD Pipfile /app/

RUN pipenv lock && pipenv install --system --deploy

ADD . /app

# Entry point
ENTRYPOINT ["/bin/bash","./docker-entrypoint.sh"]
