#!/bin/bash

set -e

docker build -f ../tensor/Dockerfile -t wombo-subnet:tensor ../
docker build -f ../neuron/Dockerfile -t wombo-subnet:neuron ../
docker build -f Dockerfile -t wombo-subnet:miner ../

docker run \
  --network="host" \
  --env-file .env \
  -v ~/.bittensor:/root/.bittensor/ \
  -it \
  --rm \
  wombo-subnet:miner \
