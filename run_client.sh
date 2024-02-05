#!/bin/bash

set -e

docker build -f client.Dockerfile -t subnet:client .

docker run \
  --network="host" \
  --gpus all \
  --env-file .env \
  -v ~/.bittensor:/root/.bittensor/ \
  -v ~/.cache:/root/.cache/ \
  subnet:client \
