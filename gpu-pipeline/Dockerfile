FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

RUN apt update && apt-get install -y build-essential

WORKDIR /app/

RUN python3 -m venv /app/venv

ENV PATH="/app/venv/bin:$PATH"

COPY ./image-generation-protocol ./image-generation-protocol/
COPY ./gpu-pipeline ./gpu-pipeline/

RUN pip install --no-cache-dir -e ./image-generation-protocol
RUN pip install --no-cache-dir -e ./gpu-pipeline
