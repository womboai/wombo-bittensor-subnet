FROM python:3.10.10-slim

WORKDIR /app/

RUN python3 -m venv ./validator/venv

ENV PATH="/app/venv:$PATH"

COPY . .

WORKDIR /app/validator/

RUN pip install -e .

EXPOSE 8091

ENTRYPOINT python \
    -m validator.main \
    --netuid $NETUID \
    --subtensor.network $NETWORK \
    --wallet.name $WALLET_NAME \
    --wallet.hotkey $WALLET_HOTKEY \
    --neuron.sample_size $SAMPLE_SIZE \
