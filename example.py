from clean_text import * 
from dataset import * 
from tokenize_data import * 

# 1. Pre clean data and dump into jsonl files

pre_clean_data(folder_path = 'data/raw/train', folder_path_out='data/interim/train')
create_jsonl_dump(
    folder_path="data/interim/train",
    out_file="books",
    path="train",
    test_size=3000,
    num_chunks=50,
)

# 2. Tokenize jsonl chunks 

NUM_CHUNKS = 50
for i in tqdm(range(NUM_CHUNKS)):
    tokenized_data = tokenize_data(dumped_file="books_train", idx=i, path="train")
    dump_into_sequences(
        file_path=f"books_train",
        tokenized_data=tokenized_data,
        idx=i,
        path="train",
    )

    tokenized_data = tokenize_data(dumped_file="books_val", idx=i, path="train")
    dump_into_sequences(
        file_path=f"books_val",
        tokenized_data=tokenized_data,
        idx=i,
        path="train",
    )