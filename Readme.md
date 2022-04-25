# CLMDataset

Small repo for creataing causal langauge modelling datasets compatible with [Webdataset](https://github.com/webdataset/webdataset).

## Steps: 
### 1. Tokenization
This assumes you have a collection of raw .txt files. ```prepare_files.py``` will clean all the text files, group them into jsonl chunks and then tokenize the chunks. Default tokenizer is GPT2's tokenizer.

```
python prepare_files.py --num-processes 20 --num-chunks 100 --validation-size 800000 --file-prefix bookcorpus
```
### 2. WebDataset Writing
```
python webdataset_writer.py --num-ctx 1024 --corpus-prefix bookcorpus
```

## Example WebDataset Use
```python
def preprocess(batch):
    x, y = (
        batch["input_id.pth"][: cfg.data.seq_len],
        batch["input_id.pth"][: cfg.data.seq_len],
    )
    return x.long(), y.long()

train_dataset = wds.DataPipeline(
        wds.SimpleShardList("data/processed/train/books_train-{000000..000013}.tar.gz"),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.shuffle(1000),
        wds.decode(),
        wds.map(preprocess),
    )
```
