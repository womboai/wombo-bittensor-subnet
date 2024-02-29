#!/bin/bash

set -e

docker --network=host build -f ../tensor/Dockerfile -t wombo-subnet:tensor ../
docker --network=host build -f ${2}Dockerfile -t $1 ../
