#!/bin/bash

python3 -m venv venv

venv/bin/pip install -e ../image-generation-protocol
venv/bin/pip install -e ../tensor --extra-index-url "https://download.pytorch.org/whl/cpu/"
venv/bin/pip install -e ../neuron
venv/bin/pip install -e .
