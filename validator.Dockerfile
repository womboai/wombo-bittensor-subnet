FROM python:3.10.10-slim

WORKDIR /app/
COPY . .

RUN python3 -m venv ./venv

ENV PATH=/app/venv:$PATH

RUN pip install -r requirements.txt

ENTRYPOINT [
    "python3",
    "-m neurons.validator",
    "--netuid $NETUID",
    "--subtensor.network $NETWORK",
    "--wallet.name $WALLET_NAME",
    "--wallet.hotkey $WALLET_HOTKEY",
    "--neurons.sample_size $SAMPLE_SIZE"
]
