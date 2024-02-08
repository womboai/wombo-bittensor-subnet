FROM python:3.10.10-slim

RUN apt update && apt-get install -y build-essential

WORKDIR /app/

RUN python3 -m venv ./client/venv

ENV PATH="/app/venv:$PATH"

COPY . .

WORKDIR /app/client/

RUN pip install -e .

EXPOSE 8000

ENTRYPOINT python \
    -m client.main \
    --netuid $NETUID \
    --subtensor.network $NETWORK \
