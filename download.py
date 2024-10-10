from huggingface_hub import snapshot_download

snapshot_download(repo_id="colbert-ir/colbert_serve",
        cache_dir=".cache",
        local_dir="data/", # should be same as the DATA_PATH in .env
        allow_patterns=["*lifestyle*/*", "*colbert*"] # this is for only downloading the lifesytle data and indexes; remove this line to download all files
)
