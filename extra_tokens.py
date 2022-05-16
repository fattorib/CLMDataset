# Adding Control Codes for Area a la https://arxiv.org/abs/1909.05858
EXTRA_TOKENS = [
    "[Abstract]",
    "[Title]",
    "Astrophysics]",
    "[High Energy Physics - Experiment]",
    "[Mathematical Physics]",
    "[Physics]",
    "[Algebraic Geometry]",
    "[Classical Analysis and ODEs]",
    "[Differential Geometry]",
    "[Dynamical Systems]",
    "[Information Theory]",
    "[Quantum Physics]",
    "[Artificial Intelligence]",
    "[Computational Engineering, Finance, and Science]",
    "[Computer Science and Game Theory]",
    "[Cryptography and Security]",
    "[Machine Learning]",
    "[Social and Information Networks]",
    "[Biomolecules]",
    "[Genomics]",
    "[Neurons and Cognition]",
    "[Machine Learning]",
    "[Statistics]",
]

from transformers import GPT2Tokenizer  


tokenizer = GPT2Tokenizer.from_pretrained("gpt2_arxiv")

with open('test.txt', 'r') as f:
    text = f.read()

encoded = tokenizer.encode(text)

print(encoded)