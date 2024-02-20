FROM wombo-subnet:tensor

RUN apt update && apt-get install -y build-essential

COPY ./client ./client/

WORKDIR /app/client/

RUN pip install -e .

EXPOSE 8000

ENTRYPOINT python \
    -m client.main \
    --logging.$LOG_LEVEL \
    --netuid $NETUID \
    --subtensor.network $NETWORK \
    --wallet.name $WALLET_NAME \
    --wallet.hotkey $WALLET_HOTKEY \
