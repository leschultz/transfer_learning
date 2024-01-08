#!/bin/bash

export PYTHONPATH=$(pwd)/../../src:$PYTHONPATH

torchrun example.py
