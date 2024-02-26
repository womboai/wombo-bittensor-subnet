#!/bin/bash

python3 -m venv venv

venv/bin/pip install -e ../image-generation-protocol
venv/bin/pip install -e ../gpu-pipeline
venv/bin/pip install -e .
