#!/bin/bash

set -e

docker build --network=host -f ../tensor/Dockerfile -t wombo-subnet:tensor ../
docker build --network=host -f ${2}Dockerfile -t $1 ../
