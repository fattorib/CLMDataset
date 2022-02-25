import jsonlines
from transformers import GPT2Tokenizer
import numpy as np
import logging
import pickle
import random
import time
from tqdm import tqdm

# Reads from a dumped jsonl file and tokenizes the data. Returns an array of shape (*, seq_len)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def tokenize_data(dumped_file, idx, path):
    """
    Takes a dumped chunk file and converts it to an array of tokens. For use in training,
    these tokens may be reshaped after.
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    bos = tokenizer.special_tokens_map["bos_token"]  # `<|endoftext|>`
    sep = tokenizer.special_tokens_map["eos_token"]  # `<|endoftext|>`

    rslt = []

    with jsonlines.open(f"data/interim/{path}/{dumped_file}_{idx}.jsonl") as reader:
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
    logger.info(f"Tokenization complete. Proceeding to reshaping and serialization.")

    return rslt_


def dump_into_sequences(file_path, tokenized_data, idx, path):
    """
    From a serialized chunk of texts (or given array), saves to .npy format. 
    """

    if tokenized_data is None:
        # From a tokenized pickel file, reshape to (*, seq_len)
        with open(f"data/interim/{path}/{file_path}_{idx}.pickle", "rb") as fp:
            bytes_array = pickle.load(fp)

    else:
        bytes_array = tokenized_data
    
    dp_file = f"data/processed/{path}/{file_path}_flattened_{idx}.npy"
    with open(dp_file, "wb") as handle:
        np.save(handle, bytes_array)


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



