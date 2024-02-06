#!/bin/bash

set -e

docker build -f Dockerfile -t wombo_subnet:validator_api ../

docker run \
  --network="host" \
  -v ~/.cache:/root/.cache/ \
  wombo_subnet:validator_api \
