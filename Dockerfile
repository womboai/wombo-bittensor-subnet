FROM nvcr.io/nvidia/tensorrt:22.11-py3

WORKDIR /app/
COPY . .

RUN apt install python3.10-venv
RUN apt install python3.10

RUN python3.10 -m venv ./venv

ENV PATH="/app/venv:$PATH"

RUN pip install -r requirements.txt
