FROM subnet:neuron

WORKDIR /app/

ENTRYPOINT python \
    -m neurons.miner \
    --netuid $NETUID \
    --subtensor.chain_endpoint $NETWORK \
    --wallet.name $WALLET_NAME \
    --wallet.hotkey $WALLET_HOTKEY \
