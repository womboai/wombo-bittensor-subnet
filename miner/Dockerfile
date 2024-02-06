FROM python:3.10.10-slim

WORKDIR /app/

RUN python3 -m venv ./miner/venv

ENV PATH="/app/venv:$PATH"

COPY tensor/requirements.txt ./tensor/requirements.txt
COPY neuron/requirements.txt ./neuron/requirements.txt
COPY miner/requirements.txt ./miner/requirements.txt

RUN pip install -r miner/requirements.txt

COPY . .

EXPOSE 8091

ENTRYPOINT python \
    -m miner.main \
    --netuid $NETUID \
    --subtensor.network $NETWORK \
    --wallet.name $WALLET_NAME \
    --wallet.hotkey $WALLET_HOTKEY \
    --generation_endpoint $GENERATION_ENDPOINT \
