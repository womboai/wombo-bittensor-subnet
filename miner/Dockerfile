FROM wombo-subnet:neuron

COPY ./miner ./miner/

WORKDIR /app/miner/

RUN pip install -e .

EXPOSE 8091

ENTRYPOINT python \
    -m miner.main \
    --netuid $NETUID \
    --subtensor.network $NETWORK \
    --wallet.name $WALLET_NAME \
    --wallet.hotkey $WALLET_HOTKEY \
    --generation_endpoint $GENERATION_ENDPOINT \
    --blacklist.force_validator_permit \
