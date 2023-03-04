import tensorflow as tf
import numpy as np
import multihead_attention
import positional_encoding


class AddNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, x, sublayer_x):
        return self.norm(x + sublayer_x)


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_ff, d_model, **kwargs):
        self.d1 = tf.keras.layers.Dense(d_ff)
        self.act1 = tf.keras.layers.ReLU()
        self.d2 = tf.keras.layers.Dense(d_model)
        super().__init__(**kwargs)

    def call(self, x):
        return self.d2(self.act1(self.d1(x)))


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, heads, d_k, d_v, d_model, d_ff, rate, **kwargs):
        super().__init__(**kwargs)
        self.multihead = multihead_attention.MultiHeadAttention(heads, d_k, d_v, d_model)
        self.drop1 = tf.keras.layers.Dropout(rate)
        self.norm1 = AddNorm()
        self.ff = FeedForward(d_ff, d_model)
        self.drop2 = tf.keras.layers.Dropout(rate)
        self.norm2 = AddNorm()

    def call(self, x, padding_mask, training):
        multihead_out = self.multihead(x, x, x, padding_mask)
        multihead_out = self.drop1(multihead_out, training=training)
        add_norm1_out = self.norm1(x, multihead_out)  # (batch, time, d_model)

        ff_out = self.ff(add_norm1_out)  # (batch, time, d_model)
        ff_out = self.drop2(ff_out, training=training)  # (batch, time, d_model)
        add_norm2_out = self.norm2(add_norm1_out, ff_out)  # (batch, time, d_model)
        return add_norm2_out


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, sequence_length, heads, d_k, d_v, d_model, d_ff, num_encoders, rate, **kwargs):
        super().__init__(**kwargs)
        self.pos_enc = positional_encoding.PositionEmbeddingFixedWeights(sequence_length, vocab_size, d_model)
        self.drop1 = tf.keras.layers.Dropout(rate)
        self.encoders = [EncoderLayer(heads, d_k, d_v, d_model, 2048, rate) for _ in range(num_encoders)]

    def call(self, input_sentence, padding_mask, training):
        encoded_sentence = self.pos_enc(input_sentence)
        x = self.drop1(encoded_sentence)

        for layer in self.encoders:
            x = layer(x, None, True)
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

    vocab_size = 20

    x = np.random.randn(batch, input_seq_len, d_model)
    n1 = AddNorm()
    print(f"{n1(x, x).shape}")

    ff = FeedForward(2048, 512)
    print(f"{ff(x).shape}")

    enc_layer = EncoderLayer(heads, d_k, d_v, d_model, 2048, drop_rate)
    print(f"{enc_layer(x, None, True).shape}")

    inp_sentence = np.random.random((batch, input_seq_len))
    pos_enc = positional_encoding.PositionEmbeddingFixedWeights(input_seq_len, vocab_size, d_model)
    pos_encoding = pos_enc(inp_sentence)
    print(f"pos enc: {pos_encoding.shape}")

    inp_sentence = np.random.randint(0, vocab_size, (batch, input_seq_len))
    encoder = Encoder(vocab_size, input_seq_len, heads, d_k, d_v, d_model, 2048, num_encoders=encoders, rate=drop_rate)
    encoder_out = encoder(inp_sentence, None, True)
    print(f"encoder output: {encoder_out.shape}")

    return


if __name__ == "__main__":
    main()
