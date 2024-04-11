from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd

if __name__ == "__main__":
    tokenizer = get_tokenizer("basic_english")
    sentences = pd.read_csv(r"data\news\forexlive.csv")["Article"].astype(str).tolist()
    vocab = build_vocab_from_iterator(map(tokenizer, sentences), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    print(vocab["<unk>"])
    # cgeck the vocab
    print(vocab(["This", "is", "the", "first", "sentence.", "this is the first sentence"]))
