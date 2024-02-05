FROM python:3.10.10-slim

RUN apt update && apt-get install -y build-essential

WORKDIR /app/

RUN python3 -m venv ./image_generator/venv

ENV PATH="/app/venv:$PATH"

COPY image_generator/requirements.txt ./image_generator/requirements.txt

RUN pip install -r image_generator/requirements.txt

COPY . .

EXPOSE 8001

ENTRYPOINT python -m image_generator.main
