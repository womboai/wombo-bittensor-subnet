#!/bin/bash

set -e

./build.sh wombo-subnet:client

docker run \
  --network="host" \
  --env-file .env \
  -v ~/.bittensor:/root/.bittensor/ \
  -it \
  --rm \
  wombo-subnet:client \
