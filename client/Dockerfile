FROM wombo-subnet:tensor

RUN apt update && apt-get install -y build-essential

COPY ./neuron-selector ./neuron-selector/
COPY ./client ./client/

RUN pip install --no-cache-dir -e ./neuron-selector

WORKDIR /app/client/

RUN pip install --no-cache-dir -e .

EXPOSE 8000

ENTRYPOINT python \
    -m client.main \
    --logging.$LOG_LEVEL \
    --netuid $NETUID \
    --subtensor.network $NETWORK \
    --wallet.name $WALLET_NAME \
    --wallet.hotkey $WALLET_HOTKEY \
