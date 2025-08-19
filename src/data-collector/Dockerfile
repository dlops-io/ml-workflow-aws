# Use the official Debian-hosted Python image
FROM python:3.11-slim

ARG DEBIAN_PACKAGES="build-essential git curl wget unzip gzip"

# Prevent apt from showing prompts
ENV DEBIAN_FRONTEND=noninteractive
#ENV CLOUDSDK_PYTHON=3.9

# Python wants UTF-8 locale
ENV LANG=C.UTF-8

# Tell pipenv where the shell is. This allows us to use "pipenv shell" as a
# container entry point.
ENV PYENV_SHELL=/bin/bash

# Tell Python to disable buffering so we don't lose any logs.
ENV PYTHONUNBUFFERED=1

# Ensure we have an up to date baseline, install dependencies and
# create a user so we don't run the app as root
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

# Install python packages
ADD Pipfile Pipfile.lock /app/

RUN pipenv install --system --deploy

# Add the rest of the source code. This is done last so we don't invalidate all
# layers when we change a line of code.
ADD . /app

# Entry point
#ENTRYPOINT ["/bin/bash"]
#ENTRYPOINT [ "pipenv","shell"]
ENTRYPOINT ["/bin/bash","./docker-entrypoint.sh"]
