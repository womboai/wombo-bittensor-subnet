FROM python:3.10.10-slim

WORKDIR /app/
COPY . .

RUN python3 -m venv ./venv

ENV PATH="/app/venv:$PATH"

RUN pip install -r requirements.txt
