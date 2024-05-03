#!/bin/bash

DIRECTORY=$(dirname $(realpath $0))

$DIRECTORY/venv/bin/python -m stress_test_validator.main $@
