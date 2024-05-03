#!/bin/bash

VALIDATOR_DIRECTORY=$(dirname $(dirname $(realpath "$(pwd)/$0")))

$VALIDATOR_DIRECTORY/venv/bin/python -m stress_test_validator.main $@
