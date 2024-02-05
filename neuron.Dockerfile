FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app/

RUN python3 -m venv ./venv

ENV PATH="/app/venv:$PATH"

COPY ./requirements.txt ./requirements.txt
COPY ./neuron.requirements.txt ./neuron.requirements.txt

RUN pip install -r neuron.requirements.txt

COPY . .
