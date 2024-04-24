#!/bin/bash

set -e

./setup.sh

venv/bin/python -m validator.main $@ &

PID=$!

echo "PID $PID"

function handle_exit() {
  kill $PID
  exit
}

trap handle_exit 0

while true; do
  sleep 1800

  # Save the current HEAD hash
  current_head=$(git rev-parse HEAD)

  git pull

  # Get the new HEAD hash
  new_head=$(git rev-parse HEAD)

  # Check if the new HEAD is different from the current HEAD
  if [ "$current_head" = "$new_head" ]; then
    continue
  fi

  # The HEAD has changed, meaning there's a new version
  echo "Validator has received an update, restarting"

  kill $PID

  ./setup.sh

  venv/bin/python -m validator.main $@ &

  PID=$!

  echo "PID $PID"
done
