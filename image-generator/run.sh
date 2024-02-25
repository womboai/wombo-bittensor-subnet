#!/bin/bash

set -e

./build.sh wombo-subnet:image-generator

docker run \
  --network="host" \
  --env-file .env \
  --gpus all \
  -v ~/.cache:/root/.cache \
  -v $(pwd)/cache:/app/image-generator/cache \
  -v $(pwd)/../checkpoints:/app/checkpoints \
  -it \
  --rm \
  wombo-subnet:image-generator \
