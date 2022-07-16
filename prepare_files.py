from multiprocessing import Pool
from functools import partial
from src.clean_text import *
from src.tokenize_data import *
from tqdm import tqdm
import os
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse():
    parser = argparse.ArgumentParser(description="Preparing CLM dataset files")

    parser.add_argument(
        "--num-processes",
        default=6, type=int
    )

    parser.add_argument(
        "--num-chunks",
        default=100, type=int
    )

    parser.add_argument(
        "--validation-size",
        default=800000, type=int
    )

    parser.add_argument(
        "--file-prefix",
        default="bookcorpus", type=str
    )
    

    args = parser.parse_args()
    return args

def main():

    args = parse()

    PROCESSES = args.num_processes
    NUM_CHUNKS = args.num_chunks
    VALIDATION_SIZE = args.validation_size
    FILE_PREFIX = args.file_prefix

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
        pool.imap(partial_validation, enumerate(validation)), total=len(validation)
    ):
        pass
    pool.close()

    # ----- 3. Tokenize Chunks ----- #

    jsonl_files_train = get_jsonl_dir(
        folder_path="data/interim/train", suffix=f"{FILE_PREFIX}_train"
    )
    tokenize_save_train = partial(
        tokenize_and_save, file_prefix=f"{FILE_PREFIX}_train", path="train"
    )
    pool = Pool(processes=PROCESSES)
    cnt = 0
    for i in tqdm(
        pool.imap(tokenize_save_train, enumerate(jsonl_files_train)),
        total=len(jsonl_files_train),
    ):
        pass
    pool.close()

    jsonl_files_val = get_jsonl_dir(
        folder_path="data/interim/train", suffix=f"{FILE_PREFIX}_val"
    )
    tokenize_save_val = partial(
        tokenize_and_save, file_prefix=f"{FILE_PREFIX}_val", path="train"
    )
    pool = Pool(processes=PROCESSES)
    cnt = 0
    for i in tqdm(
        pool.imap(tokenize_save_val, enumerate(jsonl_files_val)),
        total=len(jsonl_files_val),
    ):
        pass
    pool.close()


if __name__ == "__main__":
    main()
    

    
