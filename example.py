from clean_text import *
from dataset import *
from tokenize_data import *
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import os

if __name__ == "__main__":
    # 1. Pre clean data (multiprocessing enabled)
    PROCESSES = 6
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
    # 2. dump jsonl files

    cleaned_files_list = get_files(folder_path="data/interim/train")
    train, validation = create_train_test_split(
        cleaned_files_list, test_size=400000, num_chunks=5000
    )
    partial_train = partial(
        create_jsonl_chunked,
        folder_path="data/interim/train",
        suffix="train",
        out_file="books",
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
        out_file="books",
        path="train",
    )
    pool = Pool(processes=PROCESSES)
    cnt = 0
    for i in tqdm(
        pool.imap(partial_validation, enumerate(validation)), total=len(train)
    ):
        pass
    pool.close()

    # create_jsonl_dump(
#     folder_path="data/interim/train",
#     out_file="books",
#     path="train",
#     test_size=3000,
#     num_chunks=50,
# )


# create_jsonl_dump(
#     folder_path="data/interim/train",
#     out_file="books",
#     path="train",
#     test_size=3000,
#     num_chunks=50,
# )

# # 2. Tokenize jsonl chunks

# NUM_CHUNKS = 10000
# for i in tqdm(range(NUM_CHUNKS)):
#     tokenized_data = tokenize_data(dumped_file="books_train", idx=i, path="train")
#     dump_into_sequences(
#         file_path=f"books_train",
#         tokenized_data=tokenized_data,
#         idx=i,
#         path="train",
#     )

#     tokenized_data = tokenize_data(dumped_file="books_val", idx=i, path="train")
#     dump_into_sequences(
#         file_path=f"books_val",
#         tokenized_data=tokenized_data,
#         idx=i,
#         path="train",
#     )
