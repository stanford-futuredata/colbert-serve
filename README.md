# ColBERT-Serve

## Setup
This repo depends on the ColBERT, Pisa and Splade repositories as git submodules. Clone these repositories using the following command:
```
git submodule update --init --recursive
```
To run the code, two different conda environments are required.
```
cd ColBERT
conda env create -f conda_env_cpu.yml
conda activate colbert
pip install grpcio grpcio-tools 
```
```
cd splade_server/splade
conda create -n splade_env python=3.9
conda activate splade_env
conda env create -f conda_splade_env.yml
pip install grpcio grpcio-tools 
```
