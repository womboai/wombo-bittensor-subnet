FROM subnet:neuron

WORKDIR /app/

ENTRYPOINT python \
    -m neurons.validator \
    --netuid $NETUID \
    --subtensor.chain_endpoint $NETWORK \
    --wallet.name $WALLET_NAME \
    --wallet.hotkey $WALLET_HOTKEY \
    --neurons.sample_size $SAMPLE_SIZE \
