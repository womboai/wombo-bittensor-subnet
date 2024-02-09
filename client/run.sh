#!/bin/bash

set -e

docker build -f Dockerfile -t wombo_subnet:client ../

docker run \
  --network="host" \
  --env-file .env \
  -v ~/.bittensor:/root/.bittensor/ \
  -i \
  wombo_subnet:client \
