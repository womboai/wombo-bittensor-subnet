#!/bin/bash

set -e

docker build -f Dockerfile -t wombo_subnet:client ../

docker run \
  --network="host" \
  --env-file .env \
  -v ~/.bittensor:/root/.bittensor/ \
  --it \
  wombo_subnet:client \
