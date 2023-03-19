import pickle

import tensorflow as tf
import numpy as np
from model import TransformerModel

class Translate(tf.Module):
    def __init__(self, model, enc_seq_length, dec_seq_length, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.enc_tokenizer = self.load_tokenizer("enc")
        self.dec_tokenizer = self.load_tokenizer("dec")
        self.enc_seq_length = enc_seq_length
        self.dec_seq_length = dec_seq_length

    def load_tokenizer(self, name):
        with open(f"artifacts/tokenizer_{name}.pkl", "rb") as file:
            tokenizer = pickle.load(file)
        return tokenizer


    def pred_option1(self, enc_input):
        dec_input = []
        for i in range(enc_input.shape[0]):
            dec_input.append("<STARTT>")
        dec_input = self.dec_tokenizer.texts_to_sequences(dec_input)
        dec_input = tf.keras.preprocessing.sequence.pad_sequences(dec_input, maxlen=self.dec_seq_length, padding="post")
        dec_input = tf.convert_to_tensor(dec_input)

        inputs = (enc_input, dec_input)
        pred = self.model(inputs, training=False)
        pred = tf.argmax(pred, axis=-1)
        return pred

    def pred_option2(self, enc_input):
        start_token = ["<STARTT>"]
        start_token = self.dec_tokenizer.texts_to_sequences(start_token)
        start_token = tf.convert_to_tensor(start_token, dtype=tf.int64)
        eos_token = ["<EOS>"]
        eos_token = self.dec_tokenizer.texts_to_sequences(eos_token)
        eos_token = tf.convert_to_tensor(eos_token, dtype=tf.int64)

        dec_input = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        dec_input = dec_input.write(0, start_token)

        for i in range(self.dec_seq_length):
            inputs = (enc_input, tf.transpose(dec_input.stack())[0])
            predict = self.model(inputs, training=False)
            predict = tf.argmax(predict, axis=-1)
            dec_input = dec_input.write(i+1, predict[:, -1, tf.newaxis])

        return tf.transpose(dec_input.stack())[0]


    def Decode(self, sentences):
        result = ""
        for sentence in sentences:
            for token in sentence:
                result += self.dec_tokenizer.index_word[token] + " "
            result += " --- "
        return result


    def __call__(self, sentences):
        dec_input = []
        for i in range(len(sentences)):
            sentences[i] = "<STARTT> " + sentences[i] + " <EOS>"

        enc_input = self.enc_tokenizer.texts_to_sequences(sentences)
        enc_input = tf.keras.preprocessing.sequence.pad_sequences(enc_input, maxlen=self.enc_seq_length, padding="post")
        enc_input = tf.convert_to_tensor(enc_input)


        pred1 = self.pred_option1(enc_input)
        pred2 = self.pred_option2(enc_input)

        print(f"{enc_input=}")
        print(f"{pred1=}")
        print(f"{pred2=}")
        translation1 = self.Decode(pred1.numpy())
        translation2 = self.Decode(pred2.numpy()[:, 1:])
        print(f"{translation1=}")
        print(f"{translation2=}")

def main():
    heads = 8
    d_k = 64
    d_v = 64
    d_model = 512
    d_ff = 2048
    drop_rate = 0
    layers = 6
    enc_seq_length = 6
    enc_vocab_size = 2405
    dec_seq_length = 12
    dec_vocab_size = 3864

    model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, heads, d_k, d_v, d_model,
                             d_ff, layers, drop_rate)
    model.load_weights("model/model.chkpt")

    translate = Translate(model, enc_seq_length=enc_seq_length, dec_seq_length=dec_seq_length)

    translate(["we are in paris"])

    return

if __name__ == "__main__":
    main()
