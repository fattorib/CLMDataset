import numpy as np
import io
import webdataset as wds
from tqdm import tqdm
import torch
import json
import argparse

def parse():
    parser = argparse.ArgumentParser(description="Webdataset Writer")

    parser.add_argument(
        "--corpus-prefix",
        default=None, type=str
    )

    parser.add_argument(
        "--num-ctx",
        default=100, type=int
    )

    parser.add_argument(
        "--num-train-chunks",
        default=100, type=int
    )

    parser.add_argument(
        "--num-train-validation-chunks",
        default=100, type=int
    )

    args = parser.parse_args()
    return args

def main():

    args = parse()

    CORPUS_PREFIX = args.file_prefix
    N_CTX = args.num_ctx

    index_counter = 0

    # Writing
    with wds.ShardWriter(
        f"data/processed/sharded/{CORPUS_PREFIX}_train-%06d.tar.gz",
        maxsize=int(1e9),
        maxcount=int(1e5),
    ) as sink:
        for chunk in tqdm(range(0, args.num_train_chunks)):
            data, shape = open_chunk(
                filename=f"{CORPUS_PREFIX}_train", chunk_idx=chunk, N_CTX=N_CTX
            )
            for i, index in enumerate(data):
                sample = {
                    "__key__": f"sample_{i+index_counter}",
                    "input_id.pth": torch.tensor(index[: N_CTX]),
                }
                sink.write(sample)
            index_counter += shape[0]

    index_counter = 0
    with wds.ShardWriter(
        f"data/processed/sharded/{CORPUS_PREFIX}_val-%06d.tar.gz",
        maxsize=int(1e9),
        maxcount=int(1e5),
    ) as sink:
        for chunk in tqdm(range(0, args.num_validation_chunks)):
            data, shape = open_chunk(
                filename=f"{CORPUS_PREFIX}_val", chunk_idx=chunk, N_CTX=N_CTX
            )
            for i, index in enumerate(data):
                sample = {
                    "__key__": f"sample_{i+index_counter}",
                    "input_id.pth": torch.tensor(index[: N_CTX]),
                }
                sink.write(sample)
            index_counter += shape[0]

    # #Getting number of samples
    metadata = {}
    metadata["corpus"] = CORPUS_PREFIX
    metadata["tokenizer"] = "GPT2"
    metadata["n_ctx"] = N_CTX

    index_counter = 0
    for chunk in tqdm(range(0, args.num_train_chunks)):
        data, shape = open_chunk(
            filename=f"{CORPUS_PREFIX}_train", chunk_idx=chunk, N_CTX=N_CTX
        )
        index_counter += shape[0]

    metadata["num_train_samples"] = index_counter

    index_counter = 0
    for chunk in tqdm(range(0, args.num_validation_chunks)):
        data, shape = open_chunk(
            filename=f"{CORPUS_PREFIX}_val", chunk_idx=chunk, N_CTX=N_CTX
        )
        index_counter += shape[0]
    metadata["num_validation_samples"] = index_counter

    with open(f"data/processed/sharded/{CORPUS_PREFIX}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)



def open_chunk(filename, chunk_idx, N_CTX):
    with open(
                f"data/processed/train/{filename}_{chunk_idx}.npy",
                "rb",
        ) as f:
            buffer = io.BytesIO(f.read())
            data = np.load(buffer, allow_pickle=True)
    
    data = np.concatenate(data, dtype = np.int32)

    #Get shape of data for reshaping
    round = data.shape[0]%(N_CTX)

    data = data[:-round].reshape(-1,N_CTX)
        
    return data, data.shape


if __name__ == "__main__":
    main()
