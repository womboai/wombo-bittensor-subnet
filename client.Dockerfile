FROM python:3.10.10-slim

WORKDIR /app/
COPY . .

RUN python3 -m venv ./venv

ENV PATH="/app/venv:$PATH"

RUN pip install -r client.requirements.txt

EXPOSE 8080

ENTRYPOINT python \
    -m client \
    --netuid $NETUID \
    --subtensor.network $NETWORK \
    --wallet.name $WALLET_NAME \
    --wallet.hotkey $WALLET_HOTKEY \
