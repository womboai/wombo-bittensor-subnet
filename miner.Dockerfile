FROM subnet:neuron

WORKDIR /app/

ENTRYPOINT python \
    -m neurons.miner \
    --netuid $NETUID \
    --subtensor.network $NETWORK \
    --wallet.name $WALLET_NAME \
    --wallet.hotkey $WALLET_HOTKEY \
