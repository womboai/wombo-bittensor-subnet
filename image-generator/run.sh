#!/bin/bash

set -e

# docker build -f Dockerfile -t wombo_subnet:image_generator ../

docker run \
  --network="host" \
  --gpus all \
  -v $(pwd):/app/image-generator/ \
  -v ~/.cache:/root/.cache/ \
  wombo_subnet:image_generator \
