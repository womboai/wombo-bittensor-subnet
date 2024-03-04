#!/bin/bash

set -e

ARGUMENTS=("$@")

for i in "${!ARGUMENTS[@]}"; do
   if [[ "${ARGUMENTS[$i]}" = "--" ]]; then
       SEPARATOR_INDEX="${i}";
   fi
done

ARGUMENT_LENGTH=$(($# - $SEPARATOR_INDEX - 1))

PM2_ARGS=${@:1:$SEPARATOR_INDEX}
VALIDATOR_ARGS=${@:$(($SEPARATOR_INDEX + 2)):$ARGUMENT_LENGTH}

pm2 stop wombo-validator || true

./setup.sh

pm2 start venv/bin/python $PM2_ARGS -- -m validator.main $VALIDATOR_ARGS

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
