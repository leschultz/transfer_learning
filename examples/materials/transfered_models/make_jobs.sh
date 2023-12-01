#!/bin/bash

# The path has to be relative to where the job gets created
source_model=../../../../base_models/deltae/outputs/train/source/model.pth
layers=(0)

for layer in "${layers[@]}"
do
	for data in $(find  ../../../src/transfernet/data/ -type f -name *.csv | grep -v make_regression | awk -F '.csv' '{print $1}')
	do
		data_name=$(basename ${data})
		job_dir="jobs/frozen_${layer}_layers/${data_name}"

		mkdir -p ${job_dir}
		cp -r template/* ${job_dir}

		cd ${job_dir}

		sed -i "s/replace_freeze/${layer}/g" example.py
                sed -i "s/replace_data/${data_name}/g" example.py
                sed -i "s+replace_source_model+${source_model}+g" example.py

		cd - > /dev/null
	done
done
