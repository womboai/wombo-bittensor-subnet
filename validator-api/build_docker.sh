#!/bin/bash

set -e

docker --network=host build -f ../gpu-pipeline/Dockerfile -t wombo-subnet:gpu-pipeline ../
docker --network=host build -f Dockerfile -t $1 ../
