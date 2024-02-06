FROM python:3.10.10-slim

RUN apt update && apt-get install -y build-essential

WORKDIR /app/

RUN python3 -m venv ./validator_api/venv

ENV PATH="/app/venv:$PATH"

COPY validator_api/requirements.txt ./validator_api/requirements.txt

RUN pip install -r validator_api/requirements.txt

COPY . .

EXPOSE 8001

ENTRYPOINT python -m validator_api.main
