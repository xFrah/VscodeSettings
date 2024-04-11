def generate_embeddings(self, src):
    """
    Inputs:
        src: A list of lists of sentences. Each sub-list corresponds to a single prediction.

    Output:
        A tensor of embeddings of shape (batch_size, seq_len, d_model).
    """

    assert isinstance(src, list) and all(isinstance(s, list) for s in src), "The input must be a list of lists of sentences"

    batch_size = len(src)

    embeddings = []

    # Process each prediction separately
    for i in range(batch_size):
        # Flatten the list of sentences for one prediction into a single string
        sentences = " ".join(src[i])

        # Convert the sentences into a dictionary of tensors using the BERT tokenizer
        inputs = self._tokenizer([sentences], return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Pass the tokenized sentences through the BERT model to get the word embeddings
        x = self._bert(**inputs).last_hidden_state

        embeddings.append(x)

    # Concatenate all embeddings along the batch dimension
    embeddings = torch.cat(embeddings, dim=0)

    return embeddings
