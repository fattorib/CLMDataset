from clean_text import *
from tokenize_data import *
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import os

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    
    PROCESSES = 6
    NUM_CHUNKS = 20
    VALIDATION_SIZE = 800000 #around 10% of .txt files for openwebtext, used 3k for bookcorpus
    FILE_PREFIX = 'bookcorpus'

    # ----- 1. Preclean text ----- #
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

    # ----- 2. Dump jsonl files ----- #
    cleaned_files_list = get_files(folder_path="data/interim/train")
    train, validation = create_train_test_split(
        cleaned_files_list, test_size=VALIDATION_SIZE, num_chunks=NUM_CHUNKS
    )
    partial_train = partial(
        create_jsonl_chunked,
        folder_path="data/interim/train",
        suffix="train",
        out_file=FILE_PREFIX,
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
        out_file=FILE_PREFIX,
        path="train",
    )
    pool = Pool(processes=PROCESSES)
    cnt = 0
    for i in tqdm(
        pool.imap(partial_validation, enumerate(validation)), total=len(train)
    ):
        pass
    pool.close()

    # ----- 3. Tokenize Chunks ----- #

    jsonl_files_train = get_jsonl_dir(folder_path="data/interim/train", suffix=f'{FILE_PREFIX}_train')
    tokenize_save_train = partial(tokenize_and_save, file_prefix = f"{FILE_PREFIX}_train", path = "train")
    pool = Pool(processes=PROCESSES)
    cnt = 0
    for i in tqdm(
        pool.imap(tokenize_save_train, enumerate(jsonl_files_train)), total=len(jsonl_files_train)
    ):
        pass
    pool.close()

    jsonl_files_val = get_jsonl_dir(folder_path="data/interim/train", suffix=f'{FILE_PREFIX}_val')
    tokenize_save_val = partial(tokenize_and_save, file_prefix = f"{FILE_PREFIX}_val", path = "train")
    pool = Pool(processes=PROCESSES)
    cnt = 0
    for i in tqdm(
        pool.imap(tokenize_save_val, enumerate(jsonl_files_val)), total=len(jsonl_files_val)
    ):
        pass
    pool.close()
