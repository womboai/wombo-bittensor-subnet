#!/bin/bash

set -e

./build_docker.sh wombo-subnet:image-generator

docker run \
  --network="host" \
  --env-file .env \
  --gpus all \
  -v ~/.cache:/root/.cache \
  -v $(pwd)/cache:/app/image-generator/cache \
  -v $(pwd)/../checkpoints:/app/checkpoints \
  --detach \
  --rm \
  wombo-subnet:image-generator \
