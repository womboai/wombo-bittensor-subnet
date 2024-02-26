#!/bin/bash

set -e

docker build -f ../gpu-pipeline/Dockerfile -t wombo-subnet:gpu-pipeline ../
docker build -f ${2}Dockerfile -t $1 ../
