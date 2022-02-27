from clean_text import * 
from dataset import * 
from tokenize_data import * 
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

if __name__ == '__main__':
    # 1. Pre clean data and dump into jsonl files
    PROCESSES = 12
    files_list = get_files(folder_path='data/raw/train')

    preclean_partial = partial(pre_clean_data, folder_path_in = 'data/raw/train', folder_path_out='data/interim/train')

    pool = Pool(processes=PROCESSES)
    cnt = 0
    for i in tqdm(pool.imap(preclean_partial, files_list), total=len(files_list)):
        pass

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