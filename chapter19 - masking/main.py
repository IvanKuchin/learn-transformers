import tensorflow as tf
import numpy as np
from encoder import Encoder, EncoderLayer
from decoder import Decoder, DecoderLayer


def GetVectorizer(text, vocab_size, seq_len):
    ds = tf.data.Dataset.from_tensor_slices(text)
    vectorizer = tf.keras.layers.TextVectorization(max_tokens=vocab_size, output_sequence_length=seq_len, pad_to_max_tokens=True)
    vectorizer.adapt(text)
    print(f"vectorizer dict: {len(vectorizer.get_vocabulary())}")
    return vectorizer


class TransformerModel(tf.keras.Model):
    def __init__(self, enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, heads, d_k, d_v, d_model, d_ff_inner, layers, rate, **kwargs):
        super().__init__(**kwargs)
        self.encoder = Encoder(enc_vocab_size, enc_seq_length, heads, d_k, d_v, d_model, d_ff_inner, layers, rate)
        self.decoder = Decoder(dec_vocab_size, dec_seq_length, heads, d_k, d_v, d_model, d_ff_inner, layers, rate)
        self.dense = tf.keras.layers.Dense(dec_vocab_size)

    def PaddingMask(self, input):
        mask = tf.equal(input, 0)
        mask = tf.cast(mask, tf.float32)
        mask = mask[:, tf.newaxis, tf.newaxis, :]
        new_mask = tf.maximum(mask, tf.transpose(mask, perm=(0,1,3,2)))
        return new_mask

    def PaddingMask2(self, input):
        mask = tf.equal(input, 0)
        mask = tf.cast(mask, tf.float32)
        return mask[:, tf.newaxis, :, tf.newaxis]

    def LookaheadMask(self, seq_len):
        mask = 1-tf.linalg.band_part(tf.ones(shape=(seq_len, seq_len)), -1, 0)
        return mask

    def call(self, encoder_input, decoder_input, training):
        enc_padding_mask = self.PaddingMask(encoder_input)

        dec_padding_mask = self.PaddingMask(decoder_input)
        dec_lookahead_mask_pre = self.LookaheadMask(decoder_input.shape[-1])
        dec_lookahead_mask = tf.maximum(dec_lookahead_mask_pre, dec_padding_mask)

        dec_padding_mask2 = self.PaddingMask2(decoder_input)
        dec_lookahead_mask2 = tf.maximum(dec_lookahead_mask_pre, dec_padding_mask2)
        is_close = (tf.experimental.numpy.isclose(dec_lookahead_mask, dec_lookahead_mask2))
        is_equal = (tf.equal(dec_lookahead_mask, dec_lookahead_mask2))

        encoder_out = self.encoder(encoder_input, enc_padding_mask, training)
        decoder_out = self.decoder(decoder_input, encoder_out, dec_lookahead_mask, dec_padding_mask, training)

        dense = self.dense(decoder_out)

        return dense


def main():
    batch = 32
    input_seq_len = 9
    heads = 8
    d_k = 128
    d_v = 128
    d_model = 512
    training = True
    drop_rate = 0.1
    layers = 6
    encoder_vocab_size = 60
    decoder_vocab_size = encoder_vocab_size

    technical_phrase = "to understand machine learning algorithms you need" + \
                       " to understand concepts such as gradient of a function " + \
                       "Hessians of a matrix and optimization etc"
    wise_phrase = "patrick henry said give me liberty or give me death " + \
                  "when he addressed the second virginia convention in march"

    text = [technical_phrase, wise_phrase, "My name is Daniel", "I like to eat an icecream", "Hello Daniel", "Daniel is a bambon"]

    vectorizer = GetVectorizer(text, encoder_vocab_size, input_seq_len)
    input_seq = vectorizer(text)
    output_seq = input_seq
    # print(f"{input_seq.numpy() =}")

    model = TransformerModel(encoder_vocab_size, decoder_vocab_size, input_seq_len, input_seq_len, heads, d_k, d_v, d_model, 2048, layers, drop_rate)
    prediction = model(input_seq, output_seq, training)
    print(f"{prediction.shape=}")

    encoder_layer = EncoderLayer(heads, d_k, d_v, d_model, 2048, drop_rate)
    encoder_layer_graph = encoder_layer.build_graph()
    encoder_layer_graph.summary()

if __name__ == "__main__":
    main()
