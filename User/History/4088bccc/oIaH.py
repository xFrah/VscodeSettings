from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
import torch
from torch import Tensor
import pickle
import os


def process_sentence(raw_text_iter) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def get_vocab():
    with open(r"data\news\forexlive_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    return vocab


def save_vocab(vocab) -> None:
    with open(r"data\news\forexlive_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)


if __name__ == "__main__":
    tokenizer = get_tokenizer("basic_english")
    sentences = pd.read_csv(r"data\news\forexlive.csv")["Article"].astype(str).tolist()
    if os.path.exists(r"data\news\forexlive_vocab.pkl"):
        vocab = get_vocab()
        print("Found file forexlive_vocab.pkl, loading vocab...")
    else:
        vocab = build_vocab_from_iterator(map(tokenizer, sentences), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        print("File forexlive_vocab.pkl not found, saving new vocab...")
        save_vocab(vocab)
    a = data_process(sentences)
    print(a)
