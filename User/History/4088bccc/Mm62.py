from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
import torch
from torch import Tensor
import pickle
import os


class Vocabularizer:
    def __init__(self):
        self.tokenizer = get_tokenizer("basic_english")
        if os.path.exists(r"forexlive_vocab.pkl"):
            self.vocab = self.load_vocab()
            print("Found file forexlive_vocab.pkl, loading vocab...")
        else:
            self.vocab = build_vocab_from_iterator(map(tokenizer, sentences), specials=["<unk>"])
            self.vocab.set_default_index(vocab["<unk>"])
            print("File forexlive_vocab.pkl not found, saving new vocab...")
            self.save_vocab(vocab)

    def __call__(self, sentence) -> Tensor:
        """Converts raw text into a flat Tensor."""
        return torch.tensor(self.vocab(self.tokenizer(sentence)), dtype=torch.long)

    def build_vocab(self, sentences) -> None:
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, sentences), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])

    def load_vocab(self):
        with open(r"forexlive_vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        return vocab

    def save_vocab(self, vocab) -> None:
        with open(r"forexlive_vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)


if __name__ == "__main__":
    vocabularizer = Vocabularizer()
    sentences = pd.read_csv(r"data\news\forexlive.csv")["Article"].astype(str).tolist()
    print(sentences[0])
    a = process_sentence(sentences[0])
    print(a)
