import tensorflow as tf
import numpy as np
from multihead_attention import MultiHeadAttention
from positional_encoding import PositionEmbeddingFixedWeights
from encoder import Encoder,AddNormalization,FeedForward


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, heads, d_k, d_v, d_model, d_ff, rate, **kwargs):
        super().__init__(**kwargs)
        self.multihead1 = MultiHeadAttention(heads, d_k, d_v, d_model)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.norm1 = AddNormalization()
        self.multihead2 = MultiHeadAttention(heads, d_k, d_v, d_model)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.norm2 = AddNormalization()
        self.ff = FeedForward(d_ff, d_model)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        self.norm3 = AddNormalization()

    def call(self, x, encoder_out, lookahead_mask, padding_mask, training):
        mh1_out = self.dropout1(self.multihead1(x, x, x, lookahead_mask), training=training)
        norm1_out = self.norm1(x, mh1_out)
        mh2_out = self.dropout2(self.multihead2(norm1_out, encoder_out, encoder_out, padding_mask), training=training)
        norm2_out = self.norm2(norm1_out, mh2_out)
        ff_out = self.dropout3(self.ff(norm2_out), training=training)
        norm3_out = self.norm3(ff_out, norm2_out)
        return norm3_out


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, sequence_length, heads, d_k, d_v, d_model, d_ff, decoders, rate, **kwargs):
        super().__init__(**kwargs)
        self.pos_encoder = PositionEmbeddingFixedWeights(sequence_length, vocab_size, d_model)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.decoder_layers = [DecoderLayer(heads, d_k, d_v, d_model, d_ff, rate) for _ in range(decoders)]

    def call(self, input_sentence, encoder_out, lookahead_mask, padding_mask, training):
        x = self.dropout(self.pos_encoder(input_sentence), training=training)
        for layer in self.decoder_layers:
            x = layer(x, encoder_out, lookahead_mask, padding_mask, training)

        return x

def main():
    batch = 32
    input_seq_len = 5
    heads = 8
    d_k = 128
    d_v = 128
    d_model = 512
    training = True
    drop_rate = 0.1
    encoders = 6
    decoders = 6
    vocab_size = 20

    inputs = np.random.randint(0, vocab_size, (batch, input_seq_len))
    encoder = Encoder(vocab_size, input_seq_len, heads, d_k, d_v, d_model, 2048, encoders, drop_rate)
    enc_out = encoder(inputs, None, training)
    print(f"encoder out: {enc_out.shape}")

    # decoder_layer = DecoderLayer(heads, d_k, d_v, d_model, 2048, drop_rate)
    # dec_out = decoder_layer(np.random.randn(batch, input_seq_len, d_model), enc_out, None, None, training)
    # print(f"decoder layer out: {dec_out.shape}")

    decoder = Decoder(vocab_size, input_seq_len, heads, d_k, d_v, d_model, 2048, decoders, drop_rate)
    dec_out = decoder(inputs, enc_out, None, None, training)
    print(f"decoder out: {dec_out.shape}")

    return


if __name__ == "__main__":
    main()
