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

apt-get install redis npm
npm install -g pm2

OLD_DIRECTORY=$(pwd)
DIRECTORY=$(dirname $(realpath $0))

echo "
include /etc/redis/redis.conf
dir $DIRECTORY
" > $DIRECTORY/redis.conf

pm2 delete wombo-redis wombo-stress-test-validator wombo-forwarding-validator || true

pm2 start redis-server --name wombo-redis --interpreter none -- $DIRECTORY/redis.conf

cd $DIRECTORY/stress-test-validator
poetry install
pm2 start poetry --name wombo-stress-test-validator --interpreter none -- run python stress_test_validator/main.py $@

cd $DIRECTORY/forwarding-validator
poetry install
pm2 start poetry --name wombo-forwarding-validator --interpreter none -- run python forwarding_validator/main.py $@

pm2 save

cd $OLD_DIRECTORY
