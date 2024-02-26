#!/bin/bash

set -e

docker build -f ../tensor/Dockerfile -t wombo-subnet:tensor ../
docker build -f ${2}Dockerfile -t $1 ../
