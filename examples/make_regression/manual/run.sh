#!/bin/bash

cd create_source_model
./run.sh
cd - > /dev/null

cd scratch
./run.sh
cd - > /dev/null

cd append_model
./run.sh
cd - > /dev/null

cd freeze_n
./run.sh
cd - > /dev/null
