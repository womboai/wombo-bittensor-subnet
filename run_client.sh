#!/bin/bash

set -e

docker build -f client.Dockerfile -t subnet:client .

docker run \
  --network="host" \
  --env-file .env \
  -v ~/.bittensor:/root/.bittensor/ \
  subnet:client \
