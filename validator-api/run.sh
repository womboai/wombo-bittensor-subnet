#!/bin/bash

set -e

docker build -f ../gpu-pipeline/Dockerfile -t wombo-subnet:gpu-pipeline ../
docker build -f Dockerfile -t wombo-subnet:validator_api ../

docker run \
  --network="host" \
  --gpus=all \
  -v ~/.cache:/root/.cache/ \
  -i \
  wombo-subnet:validator_api \
