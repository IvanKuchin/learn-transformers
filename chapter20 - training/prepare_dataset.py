import pickle
import tensorflow as tf
import numpy as np

class PrepareDataset():
    def __init__(self, **kwargs):
        self.ratio = 0.9
        self.reducer = 10_000

    def create_tokenizer(self, dataset):
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(dataset)
        return tokenizer

    def get_seq_length(self, ds, tokenizer):
        encodings = tokenizer.texts_to_sequences(ds)
        lens = [len(sentence) for sentence in encodings]
        return np.max(lens)

    def tokenize_and_pad(self, tokenizer, ds, seq_len):
        ds = tokenizer.texts_to_sequences(ds)
        ds = tf.keras.preprocessing.sequence.pad_sequences(ds, maxlen=seq_len, padding="post")
        ds = tf.convert_to_tensor(ds)
        return ds

    def apply_mask(self, data, mask_length):
        mask = np.zeros((1, data.shape[-1]))
        for pos in range(mask_length):
            mask[0, pos] = 1
        masked_data = data * mask
        return masked_data

    def __call__(self, fname, dec_input_mask_length, **kwargs):
        ds = pickle.load(open(fname, "rb"))
        ds = ds[:self.reducer]
        for i in range(ds.shape[0]):
            # ds[i, 0] = "<STARTT> " + ds[i, 0] + " <EOS>"
            ds[i, 0] = "<STARTT> " + ds[i, 0] + " <EOS>"
            ds[i, 1] = "<STARTT> " + ds[i, 1] + " <EOS>"

        np.random.shuffle(ds)
        ds_X, ds_Y = ds[:, 0], ds[:, 1]

        enc_tokenizer = self.create_tokenizer(ds_X)
        enc_seq_length = self.get_seq_length(ds_X, enc_tokenizer)
        enc_vocab_size = len(enc_tokenizer.word_index) + 1

        dec_tokenizer = self.create_tokenizer(ds_Y)
        dec_seq_length = self.get_seq_length(ds_Y, dec_tokenizer) + 1  # +1 is the trick that allows remove last token in decoder_x and 1-st in decoder_Y
        dec_vocab_size = len(dec_tokenizer.word_index) + 1

        # trainX = ds_X[:int(ds_X.shape[0] * self.ratio)]
        # trainX = enc_tokenizer.texts_to_sequences(trainX)
        # trainX = tf.keras.preprocessing.sequence.pad_sequences(trainX, maxlen=enc_seq_length, padding="post")
        # trainX = tf.convert_to_tensor(trainX)

        # trainY = ds_Y[:int(ds_Y.shape[0] * self.ratio)]
        # trainY = dec_tokenizer.texts_to_sequences(trainY)
        # trainY = tf.keras.preprocessing.sequence.pad_sequences(trainY, maxlen=dec_seq_length, padding="post")
        # trainY = tf.convert_to_tensor(trainY)

        trainX_enc = self.tokenize_and_pad(enc_tokenizer, ds_X[:int(ds_X.shape[0] * self.ratio)], enc_seq_length)
        trainY_dec = self.tokenize_and_pad(dec_tokenizer, ds_Y[:int(ds_Y.shape[0] * self.ratio)], dec_seq_length)
        trainX_dec = self.apply_mask(trainY_dec, dec_input_mask_length)

        valX_enc = self.tokenize_and_pad(enc_tokenizer, ds_X[int(ds_X.shape[0] * self.ratio):], enc_seq_length)
        valY_dec = self.tokenize_and_pad(dec_tokenizer, ds_Y[int(ds_Y.shape[0] * self.ratio):], dec_seq_length)
        valX_dec = self.apply_mask(valY_dec, dec_input_mask_length)

        # when DataSet fed to train steps
        # encoder input - doesn't need STARTT-token
        # decoder input - requires STARTT-token (shift by 1 relative to encoder input) and full sentence
        # decoder output - doesn't require START-token
        return (trainX_enc[:, 1:], trainX_dec[:, :-1], trainY_dec[:, 1:], valX_enc[:, 1:], valX_dec[:, :-1], valY_dec[:, 1:], ds, enc_seq_length, dec_seq_length,
                enc_vocab_size, dec_vocab_size, enc_tokenizer, dec_tokenizer)

