import jsonlines
from matplotlib.pyplot import axes
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
    these tokens must be reshaped after.
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

    dp_file = f"data/interim/{path}/{dumped_file}_{idx}.pickle"
    vocab_size = tokenizer.vocab_size
    if vocab_size < (1 << 16):
        rslt_ = [np.uint16(d) for d in rslt]
    else:
        rslt_ = [np.int32(d) for d in rslt]
    random.shuffle(rslt_)
    logger.info(f"Tokenization complete. Proceeding to reshaping and serialization.")
    # with open(dp_file, "wb") as handle:
    #     pickle.dump(rslt_, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return rslt_


def dump_into_sequences(file_path, tokenized_data, idx, seq_len, path):
    """
    From a serialized chunk of texts (or given array), reshapes to (num_seq, seq_len). To avoid the
    case where len(serialized) % seq_len != 0, we cut off tokens to get the lengths to match.
    There may be room for a smarter approach here.
    """

    if tokenized_data is None:
        # From a tokenized pickel file, reshape to (*, seq_len)
        with open(f"data/interim/{path}/{file_path}_{idx}.pickle", "rb") as fp:
            bytes_array = pickle.load(fp)

    else:
        bytes_array = tokenized_data

    stacked_byte_arr = np.empty(shape=(1, seq_len + 1))
    for byte_array in bytes_array:
        # Remove tokens so array length is a multiple of seq_len then reshape
        length_mod_seq = byte_array.shape[0] % (seq_len + 1)
        byte_array = byte_array[: byte_array.shape[0] - length_mod_seq]
        byte_array = byte_array.reshape(-1, seq_len + 1)
        stacked_byte_arr = np.concatenate([stacked_byte_arr, byte_array], axis=0)

    dp_file = f"data/processed/{path}/{file_path}_{idx}.npy"
    with open(dp_file, "wb") as handle:
        # pickle.dump(stacked_byte_arr[1:,:], handle, protocol=pickle.HIGHEST_PROTOCOL)
        np.save(handle, stacked_byte_arr[1:, :])


if __name__ == "__main__":
    num_chunks = 50

    # dump_into_sequences(file_path=f'books_train', idx = 0,  seq_len=512, path = 'train')

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
