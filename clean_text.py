from pydoc import doc
import ftfy
import re
import os
from tqdm import tqdm
import jsonlines
import io
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


"""
Cleans, standardizes all text and parses documents into .jsonl chunks with the format:

{
    'text': *document 1 txt*,
    'text': *document 2 txt*,
    ...
    
}

"""

#GPT1 text cleaning
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


def pre_clean_data(folder_path, folder_path_out):
    """
    Pre cleans the data. Using ftfy and above text_standardize function
    """
    files = os.listdir(folder_path)

    files_list = [folder_path + f"/{f}" for f in files]

    files_out = [folder_path_out + f"/{f}" for f in files]

    for file, file_out in tqdm(zip(files_list, files_out)):
        with open(file, encoding="UTF-8") as f:
            data = text_standardize(ftfy.fix_text(f.read()))

        with open(file_out, "w", encoding="UTF-8") as f:
            f.write(data)

    logger.info("All data files have been pre-cleaned.")


def create_jsonl_dump(folder_path, out_file, num_chunks, path, test_size=1):
    """
    From a folder of cleaned files, groups them into 'num_chunks' jsonl files:
        {
            'text': *document 1 txt*,
            'text': *document 2 txt*,
            ...
        }

    'test_size' is used to control the number of files to use for validation. 
    
    TODO: No de-duplication is currently performed. To be added in the future.  
    """
    files = os.listdir(folder_path)

    files = [f for f in files if f != ".gitkeep"]

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
