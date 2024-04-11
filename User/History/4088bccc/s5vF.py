from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

if __name__ == "__main__":
    tokenizer = get_tokenizer("basic_english")
    sentences = ["This is the first sentence.", "Here is another one.", "And the final sentence."]
    vocab = build_vocab_from_iterator(map(tokenizer, sentences), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    # cgeck the vocab
    print(vocab(["This", "is", "the", "first", "sentence."]))
