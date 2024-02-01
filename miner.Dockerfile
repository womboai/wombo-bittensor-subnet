FROM subnet:neuron

WORKDIR /app/

EXPOSE 8089

ENTRYPOINT python \
    -m neurons.miner \
    --neuron.device cuda \
    --netuid $NETUID \
    --subtensor.network $NETWORK \
    --wallet.name $WALLET_NAME \
    --wallet.hotkey $WALLET_HOTKEY \
