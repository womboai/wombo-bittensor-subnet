#!/bin/bash

set -e

docker build -f Dockerfile -t wombo_subnet:$1 ../

docker run \
  --network="host" \
  --env-file .env \
  -v ~/.bittensor:/root/.bittensor/ \
  -v ~/.cache:/root/.cache/ \
  wombo_subnet:$1 \
