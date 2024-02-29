#!/bin/bash

set -e

docker build --network=host -f ../gpu-pipeline/Dockerfile -t wombo-subnet:gpu-pipeline ../
docker build --network=host -f Dockerfile -t $1 ../
