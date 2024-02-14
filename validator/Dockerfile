FROM wombo-subnet:neuron

COPY ./validator ./validator/

WORKDIR /app/validator/

RUN pip install -e .

EXPOSE 8091

ENTRYPOINT python \
    -m validator.main \
    --logging.$LOG_LEVEL \
    --netuid $NETUID \
    --subtensor.network $NETWORK \
    --wallet.name $WALLET_NAME \
    --wallet.hotkey $WALLET_HOTKEY \
    --neuron.sample_size $SAMPLE_SIZE \
