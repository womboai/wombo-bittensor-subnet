FROM subnet:neuron

WORKDIR /app/

ENTRYPOINT python \
    -m neurons.validator \
    $EXTRA \
    --netuid $NETUID \
    --subtensor.network $NETWORK \
    --wallet.name $WALLET_NAME \
    --wallet.hotkey $WALLET_HOTKEY \
    --neurons.sample_size $SAMPLE_SIZE \
