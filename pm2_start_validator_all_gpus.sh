#!/bin/bash

#
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 WOMBO
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
#
#

set -e

sudo apt-get install redis npm
sudo npm install -g pm2

PORT=$1
OLD_DIRECTORY=$(pwd)
DIRECTORY=$(dirname $(realpath $0))
GPU_COUNT=$((nvidia-smi -L || true) | wc -l)

echo "
http {
  upstream validator {
" > $DIRECTORY/nginx.conf

for i in "$(seq $GPU_COUNT)"; do
  echo "  server localhost:$(($PORT + $i));" >> $DIRECTORY/nginx.conf
done

echo "
  }

  server {
    listen $PORT http2;

    location / {
      proxy_pass grpc://validator;
    }
  }
}
" >> $DIRECTORY/nginx.conf

pm2 start nginx --name wombo-validator-nginx --interpreter none -- -c $DIRECTORY/nginx.conf

cd $DIRECTORY/stress-test
pm2 start poetry --name wombo-stress-test-validator --interpreter none -- run python stress_test_validator/main.py ${@:2}

cd $DIRECTORY/user-requests

for i in "$(seq $GPU_COUNT)"; do
  pm2 start poetry \
    --name wombo-user-requests-validator \
    --interpreter none -- \
    run python \
    user_requests_validator/main.py \
    --neuron.device "cuda:$(($i - 1))" \
    --axon.port $(($PORT + $i)) \
    --axon.external_port $PORT \
    ${@:2}
done

cd OLD_DIRECTORY
