FROM python:3.10.10-slim as neuron-image

WORKDIR /app/
COPY . .

RUN python3 -m venv ./venv

ENV PATH="/app/venv:$PATH"

RUN pip install -r requirements.txt

FROM python:3.10.10-slim

WORKDIR /app/

COPY --from=neuron-image /app/ ./

ENV PATH="/app/venv:$PATH"

ENTRYPOINT python \
    -m neurons.miner \
    --netuid $NETUID \
    --subtensor.network $NETWORK \
    --wallet.name $WALLET_NAME \
    --wallet.hotkey $WALLET_HOTKEY \
