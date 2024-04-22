#!/bin/bash

set -e

./setup.sh

venv/bin/python -m validator.main $@ &

PID=$!

echo "PID $PID"

function handle_exit() {
  kill "-$EXIT_SIGNAL" $PID
  exit
}

SIGHUP=1
SIGINT=2
SIGQUIT=3
SIGABRT=6
SIGALRM=14
SIGTERM=15

for signal in $SIGHUP $SIGINT $SIGQUIT $SIGABRT $SIGALRM $SIGTERM; do
  trap "EXIT_SIGNAL=$signal; handle_exit" $signal
done

while true; do
  sleep 1800

  # Save the current HEAD hash
  current_head=$(git rev-parse HEAD)

  # Get the new HEAD hash
  new_head=$(git rev-parse HEAD)

  # Check if the new HEAD is different from the current HEAD
  if [ "$current_head" == "$new_head" ]; then
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
