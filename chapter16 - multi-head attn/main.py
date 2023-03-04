import tensorflow as tf
import numpy as np


class DotProdAttn(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, queries, keys, values, d_k, mask=None):
        # d_k = tf.shape(keys)[-1]
        correlation_scores = tf.matmul(queries, keys, transpose_b=True)
        if mask is not None:
            correlation_scores += -1e9 * mask
        weights = tf.keras.backend.softmax(correlation_scores / tf.sqrt(tf.cast(d_k, dtype=tf.float32)))
        attn = tf.matmul(weights, values)
        return attn


class MultiHeadAttn(tf.keras.layers.Layer):
    def __init__(self, heads, d_k, d_v, d_model, **kwargs):
        super().__init__(**kwargs)
        self.heads = heads
        self.attn = DotProdAttn()
        self.d_k = d_k
        self.d_v = d_v
        self.d_o = d_model
        self.W_q = tf.keras.layers.Dense(d_k)
        self.W_k = tf.keras.layers.Dense(d_k)
        self.W_v = tf.keras.layers.Dense(d_v)
        self.W_o = tf.keras.layers.Dense(d_model)

    def reshape_tensor(self, x, heads, flag):
        if flag:
            x = tf.reshape(x, (x.shape[0], x.shape[1], heads, -1))
            x = tf.transpose(x, perm=(0, 2, 1, 3))
        else:
            x = tf.transpose(x, perm=(0, 2, 1, 3))
            x = tf.reshape(x, (x.shape[0], x.shape[1], self.d_v))
        return x

    def call(self, q, k, v, mask=None):
        reshaped_q = self.reshape_tensor(self.W_q(q), self.heads, True)
        reshaped_k = self.reshape_tensor(self.W_k(k), self.heads, True)
        reshaped_v = self.reshape_tensor(self.W_v(v), self.heads, True)

        reshaped_o = self.attn(reshaped_q, reshaped_k, reshaped_v, self.d_k / self.heads)
        o = self.reshape_tensor(reshaped_o, self.heads, False)
        return self.W_o(o)


def main():
    batch = 32
    input_seq_len = 5
    heads = 16
    d_k = 128
    d_v = 128
    d_model = 512

    q = np.random.random((batch, input_seq_len, d_k))
    k = np.random.random((batch, input_seq_len, d_k))
    v = np.random.random((batch, input_seq_len, d_v))

    multi_head_layer = MultiHeadAttn(heads, d_k, d_v, d_model)
    attn = multi_head_layer(q, k, v)
    print(f"result output: {attn.shape}")
    return


if __name__ == "__main__":
    main()
