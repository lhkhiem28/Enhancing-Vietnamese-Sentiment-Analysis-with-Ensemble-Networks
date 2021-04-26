import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer

def load_tokenizer(df_path, num_words=100000):
    train_df, test_df = pd.read_csv(df_path + "train.csv"), pd.read_csv(df_path + "test.csv")

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(list(train_df["discriptions"].astype(str).values) + list(test_df["discriptions"].astype(str).values))

    return tokenizer

def load_embedding(embedding_path, embedding_size, word_index):
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype="float32")

    embedding_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_path, encoding="utf-8", errors="ignore"))

    all_embeds = np.stack(embedding_index.values())
    embed_mean, embed_std = all_embeds.mean(), all_embeds.std()
    embedding_matrix = np.random.normal(embed_mean, embed_std, (len(word_index) + 1, embedding_size))

    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix