import pickle
import tensorflow as tf
import numpy as np

class PrepareDataset():
    def __init__(self, **kwargs):
        self.ratio_valid = 0.8
        self.ratio_test = 0.1
        self.reducer = 1_000

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

    def save_test_ds(self, ds):
        np.savetxt("artifacts/test.txt", ds, fmt="%s")

    def save_tokenizer(self, tokenizer, name):
        with open(f"artifacts/tokenizer_{name}.pkl", "wb") as file:
            pickle.dump(tokenizer, file)
        return

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
        enc_sentence_length = self.get_seq_length(ds_X, enc_tokenizer)
        enc_vocab_size = len(enc_tokenizer.word_index) + 1
        self.save_tokenizer(enc_tokenizer, "enc")

        dec_tokenizer = self.create_tokenizer(ds_Y)
        dec_sentence_length = self.get_seq_length(ds_Y, dec_tokenizer) + 1  # +1 is the trick that allows removal of the last token in decoder_input and 1-st in decoder_output
        dec_vocab_size = len(dec_tokenizer.word_index) + 1
        self.save_tokenizer(dec_tokenizer, "dec")

        # trainX = ds_X[:int(ds_X.shape[0] * self.ratio)]
        # trainX = enc_tokenizer.texts_to_sequences(trainX)
        # trainX = tf.keras.preprocessing.sequence.pad_sequences(trainX, maxlen=enc_sentence_length, padding="post")
        # trainX = tf.convert_to_tensor(trainX)

        # trainY = ds_Y[:int(ds_Y.shape[0] * self.ratio)]
        # trainY = dec_tokenizer.texts_to_sequences(trainY)
        # trainY = tf.keras.preprocessing.sequence.pad_sequences(trainY, maxlen=dec_sentence_length, padding="post")
        # trainY = tf.convert_to_tensor(trainY)

        trainX_enc = self.tokenize_and_pad(enc_tokenizer, ds_X[:int(ds_X.shape[0] * self.ratio_valid)], enc_sentence_length)
        trainY_dec = self.tokenize_and_pad(dec_tokenizer, ds_Y[:int(ds_Y.shape[0] * self.ratio_valid)], dec_sentence_length)
        trainX_dec = self.apply_mask(trainY_dec, dec_input_mask_length)

        valX_enc = self.tokenize_and_pad(enc_tokenizer, ds_X[int(ds_X.shape[0] * self.ratio_valid):int(ds_X.shape[0] * (self.ratio_valid + self.ratio_test))], enc_sentence_length)
        valY_dec = self.tokenize_and_pad(dec_tokenizer, ds_Y[int(ds_Y.shape[0] * self.ratio_valid):int(ds_Y.shape[0] * (self.ratio_valid + self.ratio_test))], dec_sentence_length)
        valX_dec = self.apply_mask(valY_dec, dec_input_mask_length)

        enc_seq_length = enc_sentence_length - 1  # remove <START>-token
        dec_seq_length = dec_sentence_length - 1  # read comment below that explains decrement

        self.save_test_ds(ds[int(ds.shape[0] * (1-self.ratio_test)):])

        # when DataSet fed to train steps
        # encoder input - doesn't need STARTT-token
        # decoder input - requires STARTT-token (shift by 1 relative to encoder input) and full sentence
        # decoder output - doesn't require START-token
        return (trainX_enc[:, 1:], trainX_dec[:, :-1], trainY_dec[:, 1:], valX_enc[:, 1:], valX_dec[:, :-1], valY_dec[:, 1:], ds, enc_seq_length, dec_seq_length,
                enc_vocab_size, dec_vocab_size, enc_tokenizer, dec_tokenizer)

