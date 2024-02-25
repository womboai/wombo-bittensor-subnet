#!/bin/bash

set -e

./build.sh wombo-subnet:validator-api

docker run \
  --network="host" \
  --env-file .env \
  --gpus=all \
  -v ~/.cache:/root/.cache \
  -v $(pwd)/cache:/app/validator-api/cache \
  -v $(pwd)/../checkpoints:/app/checkpoints \
  -it \
  --rm \
  wombo-subnet:validator-api \
