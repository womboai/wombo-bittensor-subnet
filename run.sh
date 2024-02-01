#!/bin/bash

set -e

docker build -f $1.Dockerfile -t subnet:$1 .

docker run \
  --env-file .env \
  -v ~/.bittensor:/root/.bittensor/ \
  subnet:$1
