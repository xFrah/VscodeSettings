def embed(list_of_sentences):
    # Initialize BERT model and tokenizer
    # BERT model is used to generate word embeddings.
    # BERT tokenizer is used to convert sentences into tokens that can be fed into the BERT model.
    self._bert = BertModel.from_pretrained("bert-base-uncased")
    self._tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Flatten the list of sentences for one prediction into a single string
    sentences = " ".join(src[i])

    # Convert the sentences into a dictionary of tensors using the BERT tokenizer
    inputs = self._tokenizer([sentences], return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Pass the tokenized sentences through the BERT model to get the word embeddings
    x = self._bert(**inputs).last_hidden_state

    