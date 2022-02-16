import jsonlines
from transformers import GPT2Tokenizer
import numpy as np 
import argparse
import logging
import pickle
import random
import time

#Reads from a dumped jsonl file and tokenizes the data. Returns an array of shape (*, seq_len)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


def tokenize_data(dumped_file, idx, path ):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    bos = tokenizer.special_tokens_map["bos_token"]  # `<|endoftext|>`
    sep = tokenizer.special_tokens_map["eos_token"]  # `<|endoftext|>`

    rslt = []

    with jsonlines.open(f'data/interim/{path}/{dumped_file}_{idx}.jsonl') as reader:
        for obj in reader:
            #Tokenize data    
            text = f"{bos} {obj.strip()} {sep}"
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            rslt.append(token_ids)
    
    dp_file = f"data/processed/{path}/{dumped_file}_{idx}.pickle"
    vocab_size = tokenizer.vocab_size
    if vocab_size < (1 << 16):
        rslt_ = [np.uint16(d) for d in rslt]
    else:
        rslt_ = [np.int32(d) for d in rslt]
    random.shuffle(rslt_)
    logger.info(f"Dump to {dp_file}")
    with open(dp_file, "wb") as handle:
        pickle.dump(rslt_, handle, protocol=pickle.HIGHEST_PROTOCOL)

def dump_into_sequences(file_path, idx, seq_len, path):
    #From a tokenized pickel file, reshape to (*, seq_len)
    with open(f'data/interim/{path}/{file_path}_{idx}.pickle', "rb") as fp:
        bytes_array = pickle.load(fp)

    for byte_array in bytes_array:
        # Remove tokens so array length is a multiple of seq_len then reshape 
        length_mod_seq = byte_array.shape[0] % (seq_len + 1)
        byte_array = byte_array[:byte_array.shape[0] - length_mod_seq]
        byte_array = byte_array.reshape(-1, seq_len+1)

    dp_file = f"data/processed/{path}/{file_path}_{idx}.pickle"
    with open(dp_file, "wb") as handle:
        pickle.dump(byte_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    num_chunks = 1

    for i in range(num_chunks):
        tokenize_data(dumped_file = 'harrypotter_train', idx = i, path = 'qa')

        dump_into_sequences(file_path=f'harrypotter_train', idx = i,  seq_len=512, path = 'qa')

