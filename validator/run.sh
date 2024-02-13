#!/bin/bash

set -e

docker stop wombo-validator || true

docker build -f ../tensor/Dockerfile -t wombo-subnet:tensor ../
docker build -f ../neuron/Dockerfile -t wombo-subnet:neuron ../
docker build -f Dockerfile -t wombo-subnet:validator ../

docker run \
  --network="host" \
  --env-file .env \
  -v ~/.bittensor:/root/.bittensor/ \
  --rm \
  --name wombo-validator \
  wombo-subnet:validator &

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

  docker stop wombo-validator

  if [ "$PRUNE" == "1" ]; then
    docker image prune -f
  fi

  docker build -f ../tensor/Dockerfile -t wombo-subnet:tensor ../
  docker build -f ../neuron/Dockerfile -t wombo-subnet:neuron ../
  docker build -f Dockerfile -t wombo-subnet:validator ../

  docker run \
    --network="host" \
    --env-file .env \
    -v ~/.bittensor:/root/.bittensor/ \
    --rm \
    --name wombo-validator \
    wombo-subnet:validator &
done
