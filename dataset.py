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
        base_dir,
        sample_size=0.25,
        data_path="processed",
        default_chunk_idx=0,
    ):
        """
        Approach for a CLM dataset. Assumes we are given a collection of 'chunk'
        files consisting of contiguous chunks of tokens (usually containing text for a few hundred documents).
        From the chunk, we sample 'sample_size' % of the indices to be used as starting indices for sequences. This dataset gets
        around the issue of having to pre compute chunks of a fixed size prior to training. In addition, we can
        shuffle the index data within the same chunk (on a different epoch, etc).
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.seq_len = seq_len

        self.sample_size = sample_size
        self.vocab_size = 50257
        self.token_path = tokens_name
        self.base_dir = base_dir
        self.data_path = data_path

        with open(
            f"{self.base_dir}/data/{self.data_path}/{self.token_path}_{default_chunk_idx}.npy",
            "rb",
        ) as f:
            buffer = io.BytesIO(f.read())
            data = np.load(buffer, allow_pickle=True)

        self.data = np.concatenate(data, dtype=np.int32)

        self.idx_mapping, self.len = self._create_sample_idx_mapping()

        logging.info(
            f"Using default chunk: {default_chunk_idx}. Total number of samples: {self.len}"
        )

    def _create_sample_idx_mapping(self):
        """
        from data, take a sample of indices for training
        """

        idx = np.arange(0, len(self.data) - self.seq_len)
        idx_sample = np.random.choice(
            idx, size=int((len(self.data) - self.seq_len) * self.sample_size)
        )

        return {i: idx_sample[i] for i in range(len(idx_sample))}, len(idx_sample)

    def reset_data(self, chunk_idx):
        """
        Once a chunk of data has been exhausted, update to a new chunk
        """
        try:
            with open(
                f"{self.base_dir}/data/{self.data_path}/{self.token_path}_{chunk_idx}.npy",
                "rb",
            ) as f:
                buffer = io.BytesIO(f.read())
                data = np.load(buffer, allow_pickle=True)

                del self.data, self.idx_mapping

                self.data = np.concatenate(data, dtype=np.int32)
                self.idx_mapping, self.len = self._create_sample_idx_mapping()
                logging.info(
                    f"Chunk has been updated to index {chunk_idx}. Total number of samples: {self.len}"
                )
        except Exception as e:
            logging.info(
                f"An error has occured when trying to load chunk {chunk_idx}. Reshuffling previous chunk and continuing training"
            )
            self.idx_mapping, self.len = self._create_sample_idx_mapping()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample_idx = self.idx_mapping[idx]
        sample = self.data[sample_idx : sample_idx + self.seq_len + 1]
        x = sample[:-1]
        y = sample[1:]

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
