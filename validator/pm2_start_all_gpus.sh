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
DIRECTORY=$(dirname $(realpath $0))
GPU_COUNT=$((nvidia-smi -L || true) | wc -l)

echo "
http {
  upstream validator {
" > $DIRECTORY/nginx.conf

for i in "$(seq $GPU_COUNT)"; do
  echo "  server localhost:$(($PORT + $i))" >> $DIRECTORY/nginx.conf
done

echo "
  }

  server {
    listen $PORT

    location / {
      proxy_pass http://validator
    }
  }
}
" >> $DIRECTORY/nginx.conf

pm2 start wombo-validator-nginx --name wombo-validator-nginx -- -c $DIRECTORY/nginx.conf
pm2 start $DIRECTORY/stress-test/run.sh --name wombo-stress-test-validator -- ${@:2}

for i in "$(seq $GPU_COUNT)"; do
  pm2 start \
    $DIRECTORY/user-requests/run.sh \
    --name wombo-user-requests-validator-$i -- \
    --neuron.device "cuda:$(($i - 1))" \
    --axon.port $(($PORT + $i)) \
    --axon.external_port $PORT \
    ${@:2}
done
