FROM python:3.10.10-slim

WORKDIR /app/

RUN python3 -m venv /app/venv

ENV PATH="/app/venv:$PATH"

RUN pip config set global.find-links "/app/image-generation-protocol /app/tensor"
RUN pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"

COPY ./image-generation-protocol ./image-generation-protocol/
COPY ./tensor ./tensor/

RUN pip install --no-cache-dir ./tensor
