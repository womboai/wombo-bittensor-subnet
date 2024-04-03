FROM wombo-subnet:neuron

COPY ./neuron-selector ./neuron-selector/
COPY ./validator ./validator/

RUN pip install --no-cache-dir -e ./neuron-selector

WORKDIR /app/validator/

RUN pip install --no-cache-dir -e .

ENTRYPOINT python \
    -m validator.main \
    --axon.port $AXON_PORT \
    --logging.$LOG_LEVEL \
    --neuron.device $DEVICE \
    --netuid $NETUID \
    --subtensor.network $NETWORK \
    --wallet.name $WALLET_NAME \
    --wallet.hotkey $WALLET_HOTKEY \
    $EXTRA_ARGS \
