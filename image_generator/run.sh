#!/bin/bash

set -e

docker build -f Dockerfile -t wombo_subnet:image_generator ../

docker run \
  --network="host" \
  -v ~/.cache:/root/.cache/ \
  wombo_subnet:image_generator \
