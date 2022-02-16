from email.mime import base
from torch.utils.data import Dataset
import torch

# Using Tokenizers from HF
from transformers import GPT2Tokenizer
import io
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)


class CLMDataset(Dataset):
    def __init__(
        self,
        tokens_name,
        seq_len,
        chunk_num,
        base_dir,
    ):
        """Byte-pair encoding dataset

        Args:

        """
        # TODO: Is this needed?
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.seq_len = seq_len

        self.vocab_size = 50257
        self.token_path = tokens_name
        self.base_dir = base_dir
        if chunk_num is not None:
            with open(
                f"{self.base_dir}/data/processed/{self.token_path}_{chunk_num}.npy",
                "rb",
            ) as f:
                buffer = io.BytesIO(f.read())
                data = np.load(buffer)

        logging.info(f"Total number of samples within this chunk is: {data.shape[0]}")
        self.data = data
        # del data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        dix = self.data[idx, :]
        # Data
        x = dix[:-1]

        # Targets - shifted so first k always predict k+1 token
        y = dix[1:]

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


if __name__ == "__main__":

    ds = CLMDataset(
        tokens_name="train/books_train",
        seq_len=512,
        chunk_num=0,
        base_dir="C:/Users/Ben/Desktop/CLMData",
    )

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    print(tokenizer.decode(ds[0][0]))
