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
        weights = tf.keras.backend.softmax(correlation_scores / tf.sqrt(tf.cast(d_k, dtype=tf.float64)))
        attn = tf.matmul(weights, values)
        return attn


def main():
    batch = 16
    input_seq_len = 5
    d_k = 64
    d_v = 64

    q = np.random.random((batch, input_seq_len, d_k))
    k = np.random.random((batch, input_seq_len, d_k))
    v = np.random.random((batch, input_seq_len, d_v))

    attn_layer = DotProdAttn()
    single_head = attn_layer(q, k, v, d_k)
    print(f"single head shape: {single_head.shape}")
    return


if __name__ == "__main__":
    main()