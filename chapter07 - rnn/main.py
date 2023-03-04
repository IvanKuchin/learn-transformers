import tensorflow as tf
import numpy as np
import pandas
import sklearn.preprocessing
import sklearn.metrics
import matplotlib.pyplot as plt


def CreateRNN(hidden_units, dense_units, input_shape, activations):
    inputs = tf.keras.Input(shape=input_shape)
    rnn = tf.keras.layers.SimpleRNN(hidden_units, activation=activations[0])(inputs)
    dense = tf.keras.layers.Dense(dense_units, activation=activations[1])(rnn)

    model = tf.keras.Model(inputs=inputs, outputs=dense, name="SimpleRNN")
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])
    return model


def ComputeManual(x, nn):

    print(nn.weights)

    wx = nn.get_weights()[0]
    wh = nn.get_weights()[1]
    bh = nn.get_weights()[2]
    wy = nn.get_weights()[3]
    by = nn.get_weights()[4]

    h0 = np.zeros(2).reshape((1, -1))
    h1 = x[0] @ wx + h0 @ wh + bh
    h2 = x[1] @ wx + h1 @ wh + bh
    h3 = x[2] @ wx + h2 @ wh + bh

    y = h3 @ wy + by

    # print(f"wx={wx}\nwh={wh}\nbh={bh}\nwy={wy}\nby={by}")

    return y


def ManualVSKeras():
    x = np.array([[1, 1, 1],[2, 2, 2],[3, 3, 3]])
    x = x.reshape((1, 3, -1))
    rnn = CreateRNN(2, 1, x.shape[1:], ["linear", "linear"])
    rnn.summary()

    output_rnn = rnn.predict(x)
    output_manual = ComputeManual(x[0], rnn)


    print(f"Keras  RNN: {output_rnn}")
    print(f"manual RNN: {output_manual}")
    return


def GetTrainTest(url, split_percent):
    df = pandas.read_csv(url, usecols=[1], engine="python")
    data = np.array(df.values.astype("float32"))
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0., 1.))
    data = scaler.fit_transform(data)

    split = int(len(data) * split_percent)
    train_data, test_data = data[:split], data[split:]

    return train_data, test_data, data


def MakeDS(data, time_steps):
    Y = data[range(time_steps, len(data), time_steps)]
    rows = Y.shape[0]
    x = data[:time_steps * rows]
    x = x.reshape((-1, time_steps, 1))
    return x, Y


def Graph(train_Y, train_predict, test_Y, test_predict):
    plt.plot(np.append(train_Y, test_Y))
    plt.plot(np.append(train_predict, test_predict))
    plt.legend(["Actual", "Predicted"])
    plt.axvline(len(train_Y), color="r")
    plt.show()



def ExecRNN():
    url = "dataset/monthly-sunspots.csv"
    train_data, test_data, _ = GetTrainTest(url, 0.8)
    train_x, train_Y = MakeDS(train_data, 12)
    test_x, test_Y = MakeDS(test_data, 12)

    model = CreateRNN(3, 1, train_x.shape[1:], ["tanh", "tanh"])
    model.fit(train_x, train_Y, epochs=20, validation_data=(test_x, test_Y))

    train_predict = model.predict(train_x, verbose=0)
    test_predict = model.predict(test_x, verbose=0)

    # print("mse:", sklearn.metrics.mean_squared_error(train_Y, train_predict))
    # print("mse:", sklearn.metrics.mean_squared_error(test_Y, test_predict))

    Graph(train_Y, train_predict, test_Y, test_predict)


def main():
    # ManualVSKeras()
    ExecRNN()
    return


if __name__ == '__main__':
    main()
