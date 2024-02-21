FROM wombo-subnet:neuron

COPY ./neuron-selector ./neuron-selector/
COPY ./validator ./validator/

RUN pip install --no-cache-dir -e ./neuron-selector

WORKDIR /app/validator/

RUN pip install --no-cache-dir -e .

EXPOSE 8091

ENTRYPOINT python \
    -m validator.main \
    --logging.$LOG_LEVEL \
    --netuid $NETUID \
    --subtensor.network $NETWORK \
    --wallet.name $WALLET_NAME \
    --wallet.hotkey $WALLET_HOTKEY \
    --neuron.sample_size $SAMPLE_SIZE \
