import tensorflow as tf
import numpy as np
import sklearn.preprocessing
import matplotlib.pyplot as plt


def get_fib_seq(n, scale):
    seq = np.zeros(n)
    fib_n = 0
    fib_n1 = 1
    for i in range(n):
        seq[i] = fib_n + fib_n1
        fib_n, fib_n1 = fib_n1, seq[i]
    scaler = []
    if scale:
        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
        seq = np.reshape(seq, (-1, 1))
        seq = scaler.fit_transform(seq)
        seq = np.squeeze(seq)
    return seq, scaler


def get_XY(n, time_steps, train_percent, scale):
    seq, scaler = get_fib_seq(n, scale)
    Y_idxs = np.arange(time_steps, len(seq))
    Y = seq[Y_idxs]
    x = seq[:time_steps]
    for i in range(1, Y.shape[0]):
        sub_x = seq[i:i + time_steps]
        x = np.vstack((x, sub_x))

    shuffle_idxs = np.random.permutation(Y.shape[0])
    x = x.reshape((-1, time_steps, 1))
    Y = Y.reshape((-1, 1))
    split = int(Y.shape[0] * train_percent)

    train_x, train_Y = x[shuffle_idxs[:split]], Y[shuffle_idxs[:split]]
    test_x, test_Y = x[shuffle_idxs[split:]], Y[shuffle_idxs[split:]]
    return train_x, train_Y, test_x, test_Y, scaler


def CreateRNN(input_shape, rnn_units, activations):
    inputs = tf.keras.Input(input_shape)
    rnn = tf.keras.layers.SimpleRNN(rnn_units, activation=activations[0])(inputs)
    dense = tf.keras.layers.Dense(1, activation=activations[1])(rnn)
    model = tf.keras.Model(inputs=inputs, outputs=dense, name="SimpleRNN")
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    return model


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight("attn weights", shape=(input_shape[-1], 1), initializer="random_normal", trainable=True)
        self.B = self.add_weight("attn bias", shape=(input_shape[1], 1), initializer="zeros", trainable=True)

    def call(self, x, *args, **kwargs):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.B)
        wei = tf.keras.backend.softmax(e, axis=1)
        pre_ctx = wei * x
        ctx = tf.keras.backend.sum(pre_ctx, axis=1, keepdims=False)
        return ctx

def main1():
    x = tf.constant([[[0.,1],[4,2],[3,9]],[[7,4],[2,2],[0,1]]])
    inputs = tf.keras.Input(x.shape[1:])
    att = AttentionLayer()(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=att)
    model.summary()
    ctx = model.predict(x)
    print(ctx)
    print(ctx.shape)


def AttnRNN(input_shape, rnn_units, activations):
    inputs = tf.keras.Input(input_shape)
    rnn = tf.keras.layers.SimpleRNN(rnn_units, activation=activations[0], return_sequences=True)(inputs)
    attn = AttentionLayer()(rnn)
    dense = tf.keras.layers.Dense(1, activation=activations[1])(attn)
    model = tf.keras.Model(inputs=inputs, outputs=dense)
    model.compile(loss="mse", optimizer="adam", metrics=["mse"])
    return model

def Graph(h1, h2):
    plt.plot(h1.history["mse"])
    plt.plot(h1.history["val_mse"])
    plt.plot(h2.history["mse"])
    plt.plot(h2.history["val_mse"])
    plt.legend(["SimpleRNN: Train MSE", "SimpleRNN: Validation MSE", "AttnRNN: Train MSE", "AttnRNN: Validation MSE"])
    plt.show()


def main():
    time_steps = 20
    hidden_units = 2
    epochs = 30
    batch_size = 1

    train_x, train_Y, test_x, test_Y, scaler = get_XY(1200, time_steps, 0.7, True)

    model_rnn = CreateRNN(train_x.shape[1:], hidden_units, ["tanh", "tanh"])
    model_rnn.summary()
    history1 = model_rnn.fit(train_x, train_Y, epochs=epochs, batch_size=batch_size, validation_data=(test_x, test_Y))

    model_att = AttnRNN(train_x.shape[1:], hidden_units, ["tanh", "tanh"])
    model_att.summary()
    history2 = model_att.fit(train_x, train_Y, epochs=epochs, batch_size=batch_size, validation_data=(test_x, test_Y))

    eval_model1 = model_rnn.evaluate(test_x, test_Y)[1]
    print(f'SimpleRNN validation mse: {eval_model1}')

    eval_model2 = model_att.evaluate(test_x, test_Y)[1]
    print(f'AttnRNN validation mse: {eval_model2}')
    print(model_att.get_layer("attention_layer").weights)

    Graph(history1, history2)

    return


if __name__ == '__main__':
    main()
    # main1()