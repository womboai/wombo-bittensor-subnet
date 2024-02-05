FROM python:3.10.10-slim

RUN apt update && apt-get install -y build-essential

WORKDIR /app/

RUN python3 -m venv ./venv

ENV PATH="/app/venv:$PATH"

COPY ./client.requirements.txt ./client.requirements.txt

RUN pip install -r client.requirements.txt

EXPOSE 8080

COPY . .

ENTRYPOINT python \
    -m client \
    --netuid $NETUID \
    --subtensor.network $NETWORK \
    --wallet.name $WALLET_NAME \
    --wallet.hotkey $WALLET_HOTKEY \
