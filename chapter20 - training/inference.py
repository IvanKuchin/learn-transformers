import pickle

import tensorflow as tf
import numpy as np
from model import  TransformerModel

class Translate(tf.Module):
    def __init__(self, enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, heads, d_k, d_v, d_model,
                             d_ff, layers, drop_rate, **kwargs):
        super().__init__(**kwargs)
        self.model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, heads, d_k, d_v, d_model,
                             d_ff, layers, drop_rate)
        self.enc_tokenizer = self.load_tokenizer("enc")
        self.dec_tokenizer = self.load_tokenizer("dec")
        print(self.dec_tokenizer.get_config())

    def load_tokenizer(self, name):
        with open(f"artifacts/tokenizer_{name}.pkl", "rb") as file:
            tokenizer = pickle.load(file)
        return tokenizer


def main():
    heads = 8
    d_k = 128
    d_v = 128
    d_model = 512
    d_ff = 2048
    drop_rate = 0.1
    layers = 6
    enc_seq_length = 6
    enc_vocab_size = 2405
    dec_seq_length = 12
    dec_vocab_size = 3864

    translate = Translate(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, heads, d_k, d_v, d_model,
                             d_ff, layers, drop_rate)


    return

if __name__ == "__main__":
    main()
