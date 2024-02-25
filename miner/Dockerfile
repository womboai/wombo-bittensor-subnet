FROM wombo-subnet:neuron

COPY ./miner ./miner/

WORKDIR /app/miner/

RUN pip install --no-cache-dir -e .

ENTRYPOINT python \
    -m miner.main \
    --axon.port $AXON_PORT \
    --logging.$LOG_LEVEL \
    --neuron.device $DEVICE \
    --netuid $NETUID \
    --subtensor.network $NETWORK \
    --wallet.name $WALLET_NAME \
    --wallet.hotkey $WALLET_HOTKEY \
    --generation_endpoint $GENERATION_ENDPOINT \
    --blacklist.force_validator_permit \
    $EXTRA_ARGS \
