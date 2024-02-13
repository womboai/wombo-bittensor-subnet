FROM python:3.10.10-slim

WORKDIR /app/

RUN python3 -m venv /app/venv

ENV PATH="/app/venv:$PATH"

COPY ./image-generation-protocol ./image-generation-protocol/
COPY ./tensor ./tensor/

RUN pip install ./tensor
