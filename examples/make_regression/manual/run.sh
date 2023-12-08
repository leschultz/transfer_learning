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

for i in $(ls | grep freeze)
do
	cd ${i}
	./run.sh
	cd - > /dev/null
done	
