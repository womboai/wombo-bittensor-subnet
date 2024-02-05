#!/bin/bash

set -e

docker build -f neuron.Dockerfile -t subnet:neuron .
docker build -f $1.Dockerfile -t subnet:$1 .

docker run \
  --network="host" \
  --detach \
  --gpus all \
  --env-file .env \
  -v ~/.bittensor:/root/.bittensor/ \
  -v ~/.cache:/root/.cache/ \
  --name $1 \
  subnet:$1 \
