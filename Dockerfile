FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Fix nvidia apt repo bug by deleting it's sources file, then install system packages and clean up
RUN rm /etc/apt/sources.list.d/cuda.list && \
    apt -y update && \
    apt install -y build-essential git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app/
COPY . .

RUN python3 -m venv ./venv

ENV PATH="/app/venv:$PATH"

RUN pip install -r requirements.txt
