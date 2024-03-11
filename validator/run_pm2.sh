#!/bin/bash

set -e

NAME=$1
VALIDATOR_ARGS=${@:2:$(($# - 1))}

pm2 stop wombo-validator || true

./setup.sh

pm2 start venv/bin/python --name $NAME -- -m validator.main $VALIDATOR_ARGS

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

  pm2 stop $NAME

  ./setup.sh

  pm2 restart $NAME
done
