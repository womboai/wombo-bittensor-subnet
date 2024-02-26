#!/bin/bash

set -e

pm2 stop wombo-validator

./setup.sh

pm2 start venv/bin/python --name wombo-validator -- -m validator.main $@

function handle_sigint() {
  echo "Stopping validator"
  docker stop wombo-validator
  exit
}

trap handle_sigint SIGINT

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

  pm2 stop wombo-validator

  ./setup.sh

  pm2 restart wombo-validator
done
