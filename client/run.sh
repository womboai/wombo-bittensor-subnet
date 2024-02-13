#!/bin/bash

set -e

docker build -f ../tensor/Dockerfile -t wombo-subnet:tensor ../
docker build -f Dockerfile -t wombo-subnet:client ../

docker run \
  --network="host" \
  --env-file .env \
  -v ~/.bittensor:/root/.bittensor/ \
  -it \
  --rm \
  wombo-subnet:client \
