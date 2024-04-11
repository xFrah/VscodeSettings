def embed(list_of_sentences):
    # Initialize BERT model and tokenizer
    # BERT model is used to generate word embeddings.
    # BERT tokenizer is used to convert sentences into tokens that can be fed into the BERT model.
    self._bert = BertModel.from_pretrained("bert-base-uncased")
    self._tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
