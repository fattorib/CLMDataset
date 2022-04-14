import numpy as np 
import io
import webdataset as wds
from tqdm import tqdm 
import torch 
import json 

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

if __name__ == '__main__':
    CORPUS_PRFIX = 'openwebtext'
    N_CTX = 1024
    index_counter = 0

    # Writing 
    with wds.ShardWriter(f"data/processed/sharded/{CORPUS_PRFIX}_train-%06d.tar.gz", maxsize=int(1e9), maxcount=int(1e5)) as sink:
        for chunk in tqdm(range(0,100)):
            data, shape = open_chunk(filename = f'{CORPUS_PRFIX}_train', chunk_idx = chunk, N_CTX=N_CTX)
            for i, index in enumerate(data):
                sample = {"__key__": f"sample_{i+index_counter}",
                    "input_id.pth": torch.tensor(index[:N_CTX+1])}
                sink.write(
                    sample
                )
            index_counter += shape[0]

    index_counter = 0
    with wds.ShardWriter(f"data/processed/sharded/{CORPUS_PRFIX}_val-%06d.tar.gz", maxsize=int(1e9), maxcount=int(1e5)) as sink:
        for chunk in tqdm(range(0,10)):
            data, shape = open_chunk(filename = f'{CORPUS_PRFIX}_val', chunk_idx = chunk, N_CTX=N_CTX)
            for i, index in enumerate(data):
                sample = {"__key__": f"sample_{i+index_counter}",
                    "input_id.pth": torch.tensor(index[:N_CTX+1])}
                sink.write(
                    sample
                )
            index_counter += shape[0] 

    # #Getting number of samples
    metadata = {}
    metadata['corpus'] = CORPUS_PRFIX
    metadata['tokenizer'] = 'GPT2'
    metadata['n_ctx'] = N_CTX

    index_counter = 0
    for chunk in tqdm(range(0,100)):
        data, shape = open_chunk(filename = f'{CORPUS_PRFIX}_train', chunk_idx = chunk, N_CTX=N_CTX)
        index_counter += shape[0]

    metadata['num_train_samples'] = index_counter

    index_counter = 0
    for chunk in tqdm(range(0,10)):
        data, shape = open_chunk(filename =f'{CORPUS_PRFIX}_val', chunk_idx = chunk, N_CTX=N_CTX)
        index_counter += shape[0]
    metadata['num_validation_samples'] = index_counter

    with open(f"data/processed/sharded/{CORPUS_PRFIX}_metadata.json", "w") as f:
        json.dump(metadata, f, indent = 4)
  


