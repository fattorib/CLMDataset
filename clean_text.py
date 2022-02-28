from typing import Tuple
import ftfy
import re
import os
from tqdm import tqdm
import jsonlines
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def text_standardize(text):
    """
    fixes some issues the spacy tokenizer had on books corpus
    also does some whitespace standardization
    """
    text = text.replace("—", "-")
    text = text.replace("–", "-")
    text = text.replace("―", "-")
    text = text.replace("…", "...")
    text = text.replace("´", "'")
    text = re.sub(
        """(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)""",
        r" \1 ",
        text,
    )
    text = re.sub("\s*\n\s*", " \n ", text)
    text = re.sub("\s*\t\s*", " ", text)
    text = re.sub("[^\S\n]+", " ", text)
    return text.strip()


def get_files(folder_path):
    files = os.listdir(folder_path)

    files = [f for f in files if ".txt" in f]

    return files


# takes in a list
def pre_clean_data_lst(files, folder_path_in, folder_path_out):
    """
    Pre cleans the data. Using ftfy and above text_standardize function
    """

    files_list = [folder_path_in + f"/{f}" for f in files]

    files_out = [folder_path_out + f"/{f}" for f in files]

    # for file, file_out in tqdm(zip(files_list, files_out)):
    for file, file_out in zip(files_list, files_out):
        with open(file, "r", encoding="UTF-8") as f:
            data = text_standardize(ftfy.fix_text(f.read()))

        with open(file_out, "w", encoding="UTF-8") as f:
            f.write(data)

    logger.info("All data files have been pre-cleaned.")


# takes in a single file. Useful for multiprocessing
def pre_clean_data(file, folder_path_in, folder_path_out):
    """
    Pre cleans the data. Using ftfy and above text_standardize function
    """

    file_in = folder_path_in + f"/{file}"

    file_out = folder_path_out + f"/{file}"

    with open(file_in, "r", encoding="UTF-8") as f:
        data = text_standardize(ftfy.fix_text(f.read()))

    with open(file_out, "w", encoding="UTF-8") as f:
        f.write(data)


# Takes lists as input
def create_jsonl_dump(files, folder_path, out_file, num_chunks, path, test_size=400000):
    """
    From a folder of cleaned files, groups them into 'num_chunks' jsonl files:
        {
            'text': *document 1 txt*,
            'text': *document 2 txt*,
            ...
        }

    'test_size' is used to control the number of files to use for validation.

    """

    train, val = train_test_split(files, test_size=test_size, random_state=1996)

    docs_per_chunk = len(train) // num_chunks

    for i in tqdm(range(0, num_chunks)):
        texts_arr = []
        for file in train[i * docs_per_chunk : (i + 1) * docs_per_chunk]:
            with open(folder_path + "/" + file, "r", encoding="UTF-8") as f:
                text = f.read()
            texts_arr.append(text)

        with jsonlines.open(
            f"data/interim/{path}/{out_file}_train_{i}.jsonl", mode="w"
        ) as writer:
            writer.write_all(texts_arr)

    logger.info(f"Training data has been split into {num_chunks} chunks.")

    docs_per_chunk = len(val) // num_chunks
    for i in tqdm(range(0, num_chunks)):
        texts_arr = []
        for file in val[i * docs_per_chunk : (i + 1) * docs_per_chunk]:
            with open(folder_path + "/" + file, "r", encoding="UTF-8") as f:
                text = f.read()
            texts_arr.append(text)

        with jsonlines.open(
            f"data/interim/{path}/{out_file}_val_{i}.jsonl", mode="w"
        ) as writer:
            writer.write_all(texts_arr)

    logger.info(f"Validation data has been split into {num_chunks} chunks.")


def create_train_test_split(files, test_size, num_chunks):
    """
    Takes a list of files and returns an list like:

    [[*chunk_1*], [*chunk_2*], ...]

    """
    train, val = train_test_split(files, test_size=test_size, random_state=1996)

    docs_per_chunk = len(train) // num_chunks

    train_document_list = []
    for i in range(0, num_chunks):
        train_document_list.append(train[i * docs_per_chunk : (i + 1) * docs_per_chunk])

    val_document_list = []
    for i in range(0, num_chunks):
        val_document_list.append(val[i * docs_per_chunk : (i + 1) * docs_per_chunk])

    return train_document_list, val_document_list


# Takes lists as input
def create_jsonl_chunked(file_list, folder_path, suffix, out_file, path):
    """
    Dumps list of files into a single jsonl chunk
    """
    i, data = file_list

    texts_arr = []
    for file in data:
        with open(folder_path + "/" + file, "r", encoding="UTF-8") as f:
            text = f.read()
        texts_arr.append(text)

    with jsonlines.open(
        f"data/interim/{path}/{out_file}_{suffix}_{i}.jsonl", mode="w"
    ) as writer:
        writer.write_all(texts_arr)


if __name__ == "__main__":

    files = get_files(folder_path="data/raw/train")

    a, b = create_train_test_split(files, test_size=300, num_chunks=50)
