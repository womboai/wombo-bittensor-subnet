FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app/

RUN python3 -m venv ./validator/venv

ENV PATH="/app/venv:$PATH"

COPY tensor/requirements.txt ./tensor/requirements.txt
COPY neuron/requirements.txt ./neuron/requirements.txt
COPY validator/requirements.txt ./validator/requirements.txt

RUN pip install -r validator/requirements.txt

COPY . .

EXPOSE 8091

ENTRYPOINT python \
    -m neurons.validator \
    --netuid $NETUID \
    --subtensor.network $NETWORK \
    --wallet.name $WALLET_NAME \
    --wallet.hotkey $WALLET_HOTKEY \
    --neurons.sample_size $SAMPLE_SIZE \
