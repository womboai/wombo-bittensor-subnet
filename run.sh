#!/bin/bash

set -e

docker build -f $0.Dockerfile -t subnet:$0 .

docker run --env-file .env subnet:$0
