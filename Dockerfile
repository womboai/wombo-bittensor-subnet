FROM nvcr.io/nvidia/tensorrt:22.11-py3

WORKDIR /app/
COPY . .

RUN apt-get update && apt-get install python3.8-venv

RUN python3 -m venv ./venv

ENV PATH="/app/venv:$PATH"

RUN pip install -r requirements.txt
