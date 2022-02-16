# How to handle text properly?

1. Keep it the way it is. Would need to add in a document seprarator between files. Not a huge issue to implement. The way this works however, 
we need many steps since every sentence appears multiple times. Seems in efficient.

2. Convert all data to tokens and split into chunks. Requires some care behind the scenes with dataset. Pros is we see more data for fewer steps and have to worry less about overlap between multiple files. 


## Idea:

data = {
    'text': *first document text*,
    'text: '*second document text*,
    ...
}

*convert each document to tokens*

data_tok = {
    'text: *first doc tokenized*,
    'text': *second doc tokenized*
}

*split all text tokens into smaller chunks and save* 

In the end we would be able to access the complete data array with a single index