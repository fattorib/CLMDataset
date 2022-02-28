from typing import Tuple
import jsonlines
from transformers import GPT2Tokenizer
import numpy as np
import random
from tqdm import tqdm
import os

def tokenize_data(dumped_file, path):
    """
    Takes a dumped chunk file and converts it to an array of tokens. For use in training,
    these tokens may be reshaped after.
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    bos = tokenizer.special_tokens_map["bos_token"]  # `<|endoftext|>`
    sep = tokenizer.special_tokens_map["eos_token"]  # `<|endoftext|>`

    rslt = []

    with jsonlines.open(f"data/interim/{path}/{dumped_file}") as reader:
        for obj in reader:
            # Tokenize data
            text = f"{bos} {obj.strip()} {sep}"
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            rslt.append(token_ids)

    vocab_size = tokenizer.vocab_size
    if vocab_size < (1 << 16):
        rslt_ = [np.uint16(d) for d in rslt]
    else:
        rslt_ = [np.int32(d) for d in rslt]
    random.shuffle(rslt_)

    return rslt_


def dump_into_sequences(file_path, byte_array, idx, path):
    """
    From a serialized chunk of texts (or given array), saves to .npy format.
    """

    dp_file = f"data/processed/{path}/{file_path}_{idx}.npy"
    with open(dp_file, "wb") as handle:
        np.save(handle, byte_array)

def tokenize_and_save(dumped_file, file_prefix, path):

    assert type(dumped_file) == Tuple, "Must pass an enum, not a list"

    idx, dumped_file = dumped_file
    tokenized_data = tokenize_data(dumped_file=dumped_file, path=path)

    dump_into_sequences(
            file_path=file_prefix,
            byte_array=tokenized_data,
            idx=idx,
            path="train",
        )

def get_jsonl_dir(folder_path, suffix):
    files = os.listdir(folder_path)

    files = [f for f in files if ".jsonl" in f and suffix in f]

    return files

if __name__ == "__main__":
    num_chunks = 50

    for i in tqdm(range(num_chunks)):
        tokenized_data = tokenize_data(dumped_file="books_train", idx=i, path="train")
        dump_into_sequences(
            file_path=f"books_train",
            tokenized_data=tokenized_data,
            idx=i,
            seq_len=512,
            path="train",
        )

        tokenized_data = tokenize_data(dumped_file="books_val", idx=i, path="train")
        dump_into_sequences(
            file_path=f"books_val",
            tokenized_data=tokenized_data,
            idx=i,
            seq_len=512,
            path="train",
        )
