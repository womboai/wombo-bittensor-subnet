#!/bin/bash

python3 -m venv venv

venv/bin/pip install -e ../image-generation-protocol
venv/bin/pip install -e ../tensor
venv/bin/pip install -e ../neuron-selector
venv/bin/pip install -e ../neuron
venv/bin/pip install -e .
