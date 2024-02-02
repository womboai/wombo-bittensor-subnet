#!/bin/bash

set -e

docker build -f $1.Dockerfile -t subnet:$1 .

docker run \
  --detach \
  --network="host" \
  --gpus all \
  --env-file .env \
  -v ~/.bittensor:/root/.bittensor/ \
  -v ~/.cache:/root/.cache/ \
  --name $1 \
  subnet:$1 \

docker attach $1
