#!/bin/bash

set -e

docker container rm validator

docker build -f Dockerfile -t wombo_subnet:validator ../

docker run \
  --network="host" \
  --detach \
  --env-file .env \
  -v ~/.bittensor:/root/.bittensor/ \
  --name validator \
  wombo_subnet:validator \

while true; do
  sleep 1800

  # Save the current HEAD hash
  current_head=$(git rev-parse HEAD)

  git pull

  # Get the new HEAD hash
  new_head=$(git rev-parse HEAD)

  # Check if the new HEAD is different from the current HEAD
  if [ "$current_head" == "$new_head" ]; then
    continue
  fi

  # The HEAD has changed, meaning there's a new version
  echo "Validator has received an update, restarting"

  docker stop validator

  if [ "$PRUNE" == "1" ]; then
    docker image prune -f
  fi

  docker container rm validator

  docker build -f Dockerfile -t wombo_subnet:validator ../

  docker run \
    --network="host" \
    --detach \
    --env-file .env \
    -v ~/.bittensor:/root/.bittensor/ \
    --name validator \
    wombo_subnet:validator
done
