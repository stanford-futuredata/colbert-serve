# ColBERT-Serve

## Setup
This repo depends on the ColBERT, Pisa and Splade repositories as git submodules. Clone these repositories using the following command:
```
git submodule update --init --recursive
```
To run the code, two different conda environments are required. Also the respective protobufs need to be built.
```
cd ColBERT
conda env create -f conda_env_cpu.yml
conda activate colbert
pip install grpcio grpcio-tools psutil
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. server.proto 
```
```
cd splade_server/splade
conda create -n splade_env python=3.9
conda activate splade_env
conda env create -f conda_splade_env.yml
conda activate splade
pip install grpcio grpcio-tools 
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. splade.proto 
```

### Building Pisa
```
cd pisa
mkdir build
cd build
cmake ..
make -j 
```

## Data
All necessary data must be placed in a specific folder. This folder's path must be updated in the `DATA_PATH` varriable in the `.env` file. This folder must also contain ColBERT's model checkpoint (`msmarco.psg.kldR2.nway64.ib__colbert-400000`).
The folder should be structured as follows:
```
|- <experiment_name> 
|  |- questions.tsv  (questions dataset to be run against the server)
|
|- indexes (must contain the relevant ColBERT indices which are MMAP compatible if necessary)
|  |- wiki.2018.latest.mmap
|  |- ...
|  |- ...
|
|- msmarco.psg.kldR2.nway64.ib__colbert-400000
```

The script supports the following experiments (ColBERT index name is mentioned in parantheses)
```
wiki (wiki.2018.latest<.mmap>)
msmarco (msmarco.nbits=2.latest<.mmap>)
lifestyle (lifestyle.dev.nbits=2.latest<.mmap>)
```

## Running

Start the Pisa and splade server in separate terminals. (Note: Pisa and Splade isn't required if the main driver script is being run with the `-e search` option).

### Run Pisa
```
./build/bin/serve --encoding block_simdbp --index <file.idx> --wand <file.bmw> --documents <file.docmap> --terms <file.termmap> --algorithm maxscore --scorer quantized --weighted
```
For more information on how to build a Pisa index and other information, refer to the Pisa repository.

### Run Splade
```
cd splade_server
python splade_server.py
```

### Running the main driver code
```
python driver.py -w 1 -i $index -t $input_timing_file -o $timing_output_file -r $rankings_output_file -e $exp
```

Run `python driver.py --help` for detailed explanation of all options.
For sample timing files, refer to `traces/` folder. These traces were generated using the `generate_poisson_trace.py` file. 
