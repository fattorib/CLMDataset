# CLMDataset

Small repo for creataing causal langauge modelling datasets compatible with [Webdataset](https://github.com/webdataset/webdataset).

## Steps: 
This assumes you have a collection of raw .txt files
1. (Optional) Clean all text files using ftfy:
```python
files_list = get_files(folder_path="data/raw/train")
preclean_partial = partial(
    pre_clean_data,
    folder_path_in="data/raw/train",
    folder_path_out="data/interim/train",
)

pool = Pool(processes=PROCESSES)
cnt = 0
for i in tqdm(pool.imap(preclean_partial, files_list), total=len(files_list)):
    pass

pool.close()
```

2. Dump text files into .jsonl chunks:
```python
cleaned_files_list = get_files(folder_path="data/interim/train")
train, validation = create_train_test_split(
    cleaned_files_list, test_size=VALIDATION_SIZE, num_chunks=NUM_CHUNKS
)
partial_train = partial(
    create_jsonl_chunked,
    folder_path="data/interim/train",
    suffix="train",
    out_file="bookcorpus",
    path="train",
)

pool = Pool(processes=PROCESSES)
cnt = 0
for i in tqdm(pool.imap(partial_train, enumerate(train)), total=len(train)):
    pass
pool.close()

partial_validation = partial(
    create_jsonl_chunked,
    folder_path="data/interim/train",
    suffix="val",
    out_file="bookcorpus",
    path="train",
)
pool = Pool(processes=PROCESSES)
cnt = 0
for i in tqdm(
    pool.imap(partial_validation, enumerate(validation)), total=len(train)
):
    pass
pool.close()
```

3. Tokenize chunks. By default this uses the GPT2 tokenizer from Huggingface 
```python
jsonl_files_train = get_jsonl_dir(folder_path="data/interim/train", suffix='bookcorpus_train')
tokenize_save_train = partial(tokenize_and_save, file_prefix = "bookcorpus_train", path = "train")
pool = Pool(processes=PROCESSES)
cnt = 0
for i in tqdm(
    pool.imap(tokenize_save_train, enumerate(jsonl_files_train)), total=len(jsonl_files_train)
):
    pass
pool.close()

jsonl_files_val = get_jsonl_dir(folder_path="data/interim/train", suffix='bookcorpus_val')
tokenize_save_val = partial(tokenize_and_save, file_prefix = "bookcorpus_val", path = "train")
pool = Pool(processes=PROCESSES)
cnt = 0
for i in tqdm(
    pool.imap(tokenize_save_val, enumerate(jsonl_files_val)), total=len(jsonl_files_val)
):
    pass
pool.close()
```

4. At this point, you will have a collection of ```NUM_CHUNKS``` serialized numpy files. From here, we can write these chunks of files to shards via Webdataset:
```python
def open_chunk(filename, chunk_idx, N_CTX):
    with open(
                f"data/processed/train/{filename}_{chunk_idx}.npy",
                "rb",
        ) as f:
            buffer = io.BytesIO(f.read())
            data = np.load(buffer, allow_pickle=True)
    
    data = np.concatenate(data, dtype = np.int32)

    #Get shape of data for reshaping
    round = data.shape[0]%(N_CTX+1)

    data = data[:-round].reshape(-1,N_CTX+1)
        
    return data, data.shape

with wds.ShardWriter(f"data/processed/sharded/books_train-%06d.tar.gz", maxsize=int(3e9), maxcount=int(1e5)) as sink:
        for chunk in tqdm(range(0,20)):
            data, shape = open_chunk(filename = 'bookcorpus_train', chunk_idx = chunk, N_CTX=N_CTX)
            for i, index in enumerate(data):
                sample = {"__key__": f"sample_{i+index_counter}",
                    "input_id.pth": torch.tensor(index[:N_CTX+1])}
                sink.write(
                    sample
                )
            index_counter += shape[0]

with wds.ShardWriter(f"data/processed/sharded/books_val-%06d.tar.gz", maxsize=int(3e9), maxcount=int(1e5)) as sink:
    for chunk in tqdm(range(0,5)):
        data, shape = open_chunk(filename = 'bookcorpus_val', chunk_idx = chunk, N_CTX=N_CTX)
        for i, index in enumerate(data):
            sample = {"__key__": f"sample_{i+index_counter}",
                "input_id.pth": torch.tensor(index[:N_CTX+1])}
            sink.write(
                sample
            )
        index_counter += shape[0]
```

5. Load into Webdataset! The following code should be compatible with multiprocessing/distributed training:
```python
N_CTX = 1024
def preprocess(batch):
    x,y = batch['input_id.pth'][:N_CTX], batch['input_id.pth'][1:]
    return x.long(),y.long()

train_dataset =  wds.DataPipeline(
        wds.ResampledShards('data/processed/sharded/books_train-{000000..000013}.tar.gz'),
        wds.tarfile_to_samples(),
        wds.shuffle(1000),
        wds.decode(),
        wds.map(preprocess))
x,y = next(iter(train_dataset))
```