# CLMDataset

Basic code for creating a CLM dataset. Still a WIP 

## Webtext steps:

1. Download raw file
2. Unzip (make unzip)
3. Untar all subdirs (make untarall)
4. Copy all files to a txtonly folder ```for f in */*.txt; do echo cp -t txtonly "$f"; done```
5. preclean data + create jsonl dump
6. tokenize into tokenized chunks
7. zip and upload