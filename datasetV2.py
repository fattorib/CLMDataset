from torch.utils.data import Dataset
import torch

# Using Tokenizers from HF
from transformers import GPT2Tokenizer
import io
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

# NOTE: https://github.com/pytorch/pytorch/issues/5059 PYTORCH BEHAVES WEIRD WITH MP + NP RANDOM
# FIX: https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
class CLMDataset(Dataset):
    def __init__(
        self,
        tokens_name,
        n_ctx,
        base_dir,
        data_path="processed",
        max_chunks = 3,
        max_samples = 1000
    ):
        """
        Another approach for a CLM dataset. Assumes we are given a collection of 'chunk'
        files consisting of contiguous chunks of tokens (usually containing text for a few hundred documents).
        What the __getitem__ method does here is randomly select one of the chunks. From that chunk, it selects a 
        continuous sequence of tokens from the chunk.
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.n_ctx = n_ctx

        self.vocab_size = 50257
        self.token_path = tokens_name
        self.base_dir = base_dir
        self.data_path = data_path
        self.max_chunks = max_chunks
        self.max_samples = max_samples

    #LOL
    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):

        # 1. Select a random chunk 

        rand_chunk = np.random.choice(
            self.max_chunks, size=1
        )[0]

        # 2. Open that chunk 
        with open(
            f"{self.base_dir}/data/{self.data_path}/{self.token_path}_{rand_chunk}.npy",
            "rb",
        ) as f:
            buffer = io.BytesIO(f.read())
            data = np.load(buffer, allow_pickle=True)

        data = np.concatenate(data, dtype=np.int32)

        # 3. Sample a chunk of text. Sample is done so we have a full context
        idx = np.arange(0, len(data) - self.n_ctx)
        sample_idx = np.random.choice(
            idx,
        )
        sample = data[sample_idx : sample_idx + self.n_ctx + 1]
        x = sample[:-1]
        y = sample[1:]

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

if __name__ == '__main__':

    V2_dataset = CLMDataset(
        tokens_name= 'books_train',
        n_ctx = 1024,
        base_dir='C:/Users/Ben/Desktop/CLMData',
        data_path= 'processed/train',
        max_chunks=28
    )

    