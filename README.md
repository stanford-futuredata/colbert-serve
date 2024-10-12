# ColBERT-Serve

## Setup
This repo depends on the ColBERT, Pisa and Splade repositories as git submodules. Clone these repositories using the following command:
```
git submodule update --init --recursive
```
To run the code, two different conda environments are required. Also the respective protobufs need to be built. If you don't have conda, follow the official [conda installation guide](https://docs.anaconda.com/anaconda/install/linux/#installation).
```bash
cd ColBERT
conda env create -f conda_env_cpu.yml
conda activate colbert
pip install grpcio grpcio-tools psutil
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd ..
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. server.proto 
```
```bash
cd splade_server/splade
conda env create -f conda_splade_env.yml
conda activate splade
pip install grpcio grpcio-tools
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd .. 
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. splade.proto 
```

### Building Pisa
```bash
cd ../pisa
mkdir build
cd build
cmake ..
make -j 
```

## Overview
To repro the paper's results, go to the [Data](#data) section and directly download the model, data, and indexes. To deal with your own corpus, follow the below steps.

Step 0: Proprocess your data. The data should be tab-seperated (TSV) files: (1) `questions.tsv` that contains searching queries, where each line is `qid \t query text`; and (2) `collections.esv` that contains all passages, where each line is `pid \t passage text`.

Step 1: Download ColBERT model (`colbertv2.0` which can be downloaded [here](https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz)).

Step 2: [Index with ColBERT](#colbert), then, create the MMAP index.

Step 3: [Index with Pisa](#pisa).

Check: All necessary data must be placed in a specific folder. This folder's path must be updated in the `DATA_PATH` varriable (by default is `./data`) in the `.env` file. This folder must also contain ColBERT's model checkpoint.

The folder should be structured as follows:
```
|- <DATA_PATH>
    |- <dataset_name> 
    |  |- questions.tsv  (questions dataset to be run against the server)
    |
    |- indexes (must contain the relevant ColBERT indices which are MMAP compatible if necessary)
    |  |- wiki.2018.latest.mmap
    |  |- ...
    |  |- ...
    |
    |- colbertv2.0
```
The script supports the following `dataset_name` (ColBERT index name is mentioned in parantheses)
```
wiki (wiki.2018.latest<.mmap>)
msmarco (msmarco.nbits=2.latest<.mmap>)
lifestyle (lifestyle.dev.nbits=2.latest<.mmap>)
```

Step 4: Follow the [Running](#running) section. Run the Pisa and Splade servers; then, run the main driver to get the latency and ranking results.

## Data
All data used in this paper can be downloaded from [HuggingFace Hub](https://huggingface.co/colbert-ir/colbert_serve/tree/main). The folders and files are already formatted and placed in the right place, and are ready to use.

Use the example script to download `lifestyle` data and indexes:
```bash
pip install huggingface_hub
python download.py
```

Refer to the [HuggingFace documents](https://huggingface.co/docs/huggingface_hub/v0.25.2/en/package_reference/file_download#huggingface_hub.snapshot_download) for more instructions about downloading from HuggingFace Hub. 


## Indexing
If you want to index your own corpus, follow the below sections.

### ColBERT
First, note that a **GPU** is required for indexing. Also, the ColBERT GPU environment needs to be installed for indexing:
```bash
cd ColBERT
conda env create -f conda_env.yml -n colbert_gpu
conda activate colbert_gpu
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

Then, follow the [ColBERT repo instructions](https://github.com/stanford-futuredata/ColBERT/tree/main?tab=readme-ov-file#indexing) to index the corpus.

Last, to create the MMAP version of ColBERT index, run:
```bash
python -m colbert.utils.coalesce --input=/path/to/index/ --output=/path/to/index.mmap/
```

### Pisa
Follow the [instructions](https://gist.github.com/saarthaks/4ef756f345b478d7e96b107dea506217) to create the Pisa index using the Splade model.




## Running
From the `colbert-serve` directory:

### Prerequisite: Run Pisa and Splade

`conda activate splade`, then, run the following code to start the Pisa and Splade servers in separate terminals. (Note: Pisa and Splade isn't required if the main driver script is being run with the `-e search` option).

```bash
conda activate splade

# Pisa:
./pisa/build/bin/serve --encoding block_simdbp --index <file.idx> --wand <file.bmw> --documents <file.docmap> --terms <file.termmap> --algorithm maxscore --scorer quantized --weighted
## example:
## ./pisa/build/bin/serve --encoding block_simdbp --index data/lotte-lifestyle-pisa-index/lotte-lifestyle-index.block_simdbp.idx --wand data/lotte-lifestyle-pisa-index/lotte-lifestyle-index.fixed-40.bmw --documents data/lotte-lifestyle-pisa-index/lotte-lifestyle-index.docmap --terms data/lotte-lifestyle-pisa-index/lotte-lifestyle-index.termmap --algorithm maxscore --scorer quantized --weighted

# Splade:
## open a separate terminal
cd splade_server
python splade_server.py
```

### Running the main driver code
```bash
conda activate colbert
python driver.py -w 1 -i $index -e $exp -t $input_timing_file -o $timing_output_file -r $rankings_output_file -m
```
- `index` used in this paper are: "wiki", "msmarco", and "lifestyle". If you indexed your own corpus, set it as your own `dataset_name`.
- Supported `exp` are: "search" (full ColBERTv2), "pisa" (SPLADEv2), and "serve" (Rerank/Hybrid). 
- For sample timing files (`$input_timing_file`), refer to `traces/` folder. These traces were generated using the `generate_poisson_trace.py` file. 
- Latency output is stored in `$timing_output_file`, and ranking output is stored in `$rankings_output_file`.
- Remove `-m` to disable memory mapping.

Run `python driver.py --help` for detailed explanation of all options.

### Hosting your personal server
To host your own server, run the following command:
```bash
conda activate colbert
python server.py -w {num_torch_threads} -i $index [-m]
```

Example snippet for querying the server:
```python
import grpc
import server_pb2
import server_pb2_grpc

stub = server_pb2_grpc.ServerStub(grpc.insecure_channel('localhost:50050'))
q = server_pb2.Query(query="Who is the president of the US?", qid=1000, k=100)

print(stub.Serve(q))
```
