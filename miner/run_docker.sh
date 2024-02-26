#!/bin/bash

set -e

./build_docker.sh wombo-subnet:miner

docker run \
  --network="host" \
  --env-file .env \
  -v ~/.bittensor:/root/.bittensor/ \
  -v ~/.aws:/root/.aws/ \
  -it \
  --rm \
  wombo-subnet:miner \
