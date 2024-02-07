FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

RUN apt update && apt-get install -y build-essential

WORKDIR /app/

RUN python3 -m venv ./image_generator/venv

ENV PATH="/app/venv:$PATH"

COPY . .

WORKDIR /app/image-generator/

RUN pip install -e .

EXPOSE 8001

ENTRYPOINT python -m image_generator.main
