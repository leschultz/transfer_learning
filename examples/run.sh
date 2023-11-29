#!/bin/bash

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ./env

export PATH=$(pwd)/env/bin/:$PATH

pip install -U pip
pip install pymatgen scikit-learn matplotlib
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

tar xvf transfer_learning.tar
rm transfer_learning.tar

cd transfer_learning/scripts
python3 train.py

cd ../../

tar czf transfer_learning.tar transfer_learning
