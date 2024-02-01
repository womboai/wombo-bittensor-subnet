FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /app/
COPY . .

RUN python3 -m venv ./venv

ENV PATH="/app/venv:$PATH"

RUN pip install -r requirements.txt
