import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


vocab_size = 200
embed_dims = 50
output_seq_length = 20

def VectorizeWords(words):
    ds = tf.data.Dataset.from_tensor_slices(words)
    vec = tf.keras.layers.TextVectorization(output_sequence_length=output_seq_length, max_tokens=vocab_size)
    vec.adapt(ds)
    print(f'vocabulary: {vec.get_vocabulary()}')
    vectorized = vec(words)
    return vectorized, vec


def EmbedWords(vectorized_words):
    emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dims)
    return emb(vectorized_words), emb


def EmbedPosEnc():
    emb = tf.keras.layers.Embedding(input_dim=output_seq_length, output_dim=embed_dims)
    positions = tf.range(0, output_seq_length)
    positions = tf.expand_dims(positions, axis=0)
    return emb(positions), emb


class PositionEmbeddingFixedLayer(tf.keras.layers.Layer):
    def __init__(self, seq_len, vocab_size, emb_dims, **kwargs):
        super().__init__(**kwargs)
        word_enc_matrix = self.getPositionalEnconding(seq_len=vocab_size, d=emb_dims, n=10000)
        pos_enc_matrix = self.getPositionalEnconding(seq_len=seq_len, d=emb_dims, n=10000)
        self.word_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=emb_dims)
        self.pos_enc = tf.keras.layers.Embedding(input_dim=seq_len, output_dim=emb_dims, weights=[pos_enc_matrix], trainable=False)

    def getPositionalEnconding(self, seq_len, d, n):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in range(int(d / 2)):
                denominator = np.power(n, 2 * i / d)
                P[k, 2*i] = np.sin(k / denominator)
                P[k, 2*i+1] = np.cos(k / denominator)
        return P

    def __call__(self, inputs):
        pos_idxs = tf.range(0, inputs.shape[-1])
        pos_emb = self.pos_enc(pos_idxs)
        word_emb = self.word_emb(inputs)
        return word_emb + pos_emb


class PositionEmbeddingRandomLayer(tf.keras.layers.Layer):
    def __init__(self, seq_len, vocab_size, emb_dims, **kwargs):
        super().__init__(**kwargs)
        self.word_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=emb_dims)
        self.pos_enc = tf.keras.layers.Embedding(input_dim=seq_len, output_dim=emb_dims)

    def __call__(self, inputs):
        pos_idxs = tf.range(0, inputs.shape[-1])
        pos_emb = self.pos_enc(pos_idxs)
        word_emb = self.word_emb(inputs)
        return word_emb + pos_emb


def main1():
    vectorized_words, vectorizer_layer = VectorizeWords([["I am a robot."], ["You are a robot too"]])
    print(f"vectorized words: {vectorized_words}")
    word_emb, word_emb_layer  = EmbedWords(vectorized_words)
    print(f"word embeddings: {word_emb.shape}")
    word_pos_enc, _ = EmbedPosEnc()
    print(f"positional encodings: {word_pos_enc.shape}")
    # input_seq = tf.keras.layers.Add()([word_emb, word_pos_enc])
    input_seq = word_emb + word_pos_enc
    print(f"input seq: {input_seq.shape}")


def Graph(arr1, arr2):
    arr = np.vstack([arr1.numpy(), arr2.numpy()])
    print(arr.shape)
    fig = plt.figure(figsize=(15,10))
    for i in range(arr.shape[0]):
        ax = plt.subplot(2, 2, i+1)
        ax.matshow(arr[i])
    plt.show()


def PrintStat(arr):
    print(f"{arr.shape}  (min/mean/std/max: {tf.reduce_min(arr)}/{tf.reduce_mean(arr)}/{tf.math.reduce_std(arr)}/{tf.reduce_max(arr)})")


def main2():
    technical_phrase = "to understand machine learning algorithms you need" + \
                       " to understand concepts such as gradient of a function " + \
                       "Hessians of a matrix and optimization etc"
    wise_phrase = "patrick henry said give me liberty or give me death " + \
                  "when he addressed the second virginia convention in march"
    vectorized_words, _ = VectorizeWords([[technical_phrase], [wise_phrase]])
    print(f"vectorized words: {vectorized_words.shape}")
    pos_emb_fix_layer = PositionEmbeddingFixedLayer(seq_len=output_seq_length, vocab_size=vocab_size, emb_dims=embed_dims)
    print(f"fixed word emb and pos encoding:  ", end="")
    word_and_pos_fix = pos_emb_fix_layer(vectorized_words)
    PrintStat(word_and_pos_fix)
    pos_emb_rnd_layer = PositionEmbeddingRandomLayer(seq_len=output_seq_length, vocab_size=vocab_size, emb_dims=embed_dims)
    print(f"random word emb and pos encoding: ", end="")
    word_and_pos_rnd = pos_emb_rnd_layer(vectorized_words)
    PrintStat(word_and_pos_rnd)
    Graph(word_and_pos_rnd, word_and_pos_fix)


if __name__ == '__main__':
    main2()


