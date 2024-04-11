from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
import torch
from torch import Tensor
import pickle


def data_process(raw_text_iter) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def get_vocab_plus_tokenizer() -> tuple:
    with open(r"data\news\forexlive_vocab_plus_tokenizer.pkl", "rb") as f:
        vocab, tokenizer = pickle.load(f)
    return vocab, tokenizer


def save_vocab_plus_tokenizer(vocab, tokenizer) -> None:
    with open(r"data\news\forexlive_vocab_plus_tokenizer.pkl", "wb") as f:
        pickle.dump((vocab, tokenizer), f)


if __name__ == "__main__":
    tokenizer = get_tokenizer("basic_english")
    sentences = pd.read_csv(r"data\news\forexlive.csv")["Article"].astype(str).tolist()
    vocab = build_vocab_from_iterator(map(tokenizer, sentences), specials=["<unk>"])
    vocab_plus_tokenizer = (vocab, tokenizer)
    save_vocab_plus_tokenizer(vocab, tokenizer)
    vocab.set_default_index(vocab["<unk>"])
    a = data_process(sentences)
    print(a)
