import datetime
import os.path
import time

import tensorflow as tf
# import tensorflow_cloud as tfc
# import numpy as np
from prepare_dataset import PrepareDataset
from model import TransformerModel


class LRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def __call__(self, step, **kwargs):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = step ** -0.5
        arg2 = step * (self.warmup_steps ** -1.5)
        lr = (self.d_model ** -0.5) * tf.math.minimum(arg1, arg2)
        # if (step % 1000) == 0:
        #     print(f"lr scheduler: {step}")
        return lr


# prediction.shape is (batch, seq_len, vocab_size)
# target.shape is (batch, seq_len)
def loss_fcn(target, prediction):
    mask = tf.equal(target, 0)
    inv_mask = tf.logical_not(mask)
    inv_mask = tf.cast(inv_mask, dtype=tf.float32)

    loss = tf.keras.losses.sparse_categorical_crossentropy(target, prediction, from_logits=True)
    loss *= inv_mask
    return tf.reduce_sum(loss) / tf.reduce_sum(inv_mask)


def accuracy_fcn(target, prediction):
    mask = tf.equal(target, 0)
    inv_mask = tf.logical_not(mask)

    pred_cat = tf.math.argmax(prediction, axis=-1)
    pred_cat = tf.cast(pred_cat, dtype=tf.float32)
    accuracy = tf.equal(target, pred_cat)

    accuracy = tf.logical_and(accuracy, inv_mask)

    inv_mask = tf.cast(inv_mask, dtype=tf.float32)
    accuracy = tf.cast(accuracy, dtype=tf.float32)

    return tf.reduce_sum(accuracy) / tf.reduce_sum(inv_mask)


# @tf.function
def train_step(model, optimizer, enc_x, dec_x, dec_Y):
    with tf.GradientTape() as tape:
        pred = model((enc_x, dec_x), training=True)
        loss = loss_fcn(dec_Y, pred)

    grad = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grad, model.trainable_weights))
    accuracy = accuracy_fcn(dec_Y, pred)
    return loss, accuracy


def test_loss_and_accuracy():
    print("--- test_loss_and_accuracy")
    y_pred = tf.constant([[[0, 0, 0, 0.05, 0.95], [0, 0, 0.8, 0.1, 0.1], [0, 0.95, 0, 0.05, 0], [0, 0, 0.1, 0.8, 0.1],
                           [0.8, 0, 0.1, 0, 0.1], [0.6, 0.1, 0.1, 0.1, 0.1]]])
    y_true = tf.constant([[4, 2, 1, 3, 0, 0]])  # matching to y_pred
    # y_true = tf.constant([[4, 2, 4, 3, 0, 0]])  # no match to y_pred
    print(f"{loss_fcn(y_true, y_pred)=}")
    print(f"{accuracy_fcn(y_true, y_pred)=}")
    return


def test_lr_scheduler(d_model):
    print("--- test_lr_scheduler")
    lr_schedule = LRScheduler(d_model, 4000)
    for step in range(1, 40000, 1000):
        print(f"{step:6}: {lr_schedule(step)}")
    return


def test_ds(ds):
    print("--- test_ds")
    for i, entry in enumerate(ds):
        (x1, x2), Y = entry
        print(f"{i}) {x1.shape=}, {x2.shape=}, {Y.shape=}")


def make_a_prediction(message, model, valX_enc, valX_dec, valY_dec):
    print(message)
    preds = model.predict((valX_enc[0:1], valX_dec[0:1]))
    print(valX_enc[0:1].numpy())
    print(valX_dec[0:1].numpy())
    print(valY_dec[0:1].numpy())
    print(tf.argmax(preds, axis=-1))



def PrepActions(dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)


def main():
    batch = 64
    heads = 8
    d_k = 64
    d_v = 64
    d_model = 512
    d_ff = 2048
    drop_rate = 0.1
    layers = 6
    epochs = 2

    is_remote = (os.environ.get("TF_KERAS_RUNNING_REMOTELY") == "1")  # tfc.remote()
    fit_verbosity = 1
    checkpoint_path = ""
    tensorboard_path = ""
    model_save_path = ""

    PrepActions("artifacts")

    prep = PrepareDataset()
    trainX_enc, trainX_dec, trainY_dec, valX_enc, valX_dec, valY_dec, ds, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size, enc_tokenizer \
        , dec_tokenizer = prep("dataset/english-german-both.pkl")

    print(
        f"train x: {trainX_enc.shape}\ntrain Y: {trainY_dec.shape}\nvalidation x: {valX_enc.shape}\nvalidation Y: {valY_dec.shape}")
    print(f"{enc_seq_length=}\n{enc_vocab_size=}\n{dec_seq_length=}\n{dec_vocab_size=}")
    # test_loss_and_accuracy()
    # test_lr_scheduler(d_model)

    options = tf.data.Options()
    options.deterministic = False
    train_ds = tf.data.Dataset.from_tensor_slices(((trainX_enc, trainX_dec), trainY_dec)).batch(4 * batch).prefetch(tf.data.experimental.AUTOTUNE)
    valid_ds = tf.data.Dataset.from_tensor_slices(((valX_enc, valX_dec), valY_dec)).batch(8 * batch).prefetch(tf.data.experimental.AUTOTUNE)
    # test_ds(train_ds)

    # strategy = tf.distribute.MirroredStrategy()
    # train_ds = strategy.experimental_distribute_dataset(train_ds)
    # valid_ds = strategy.experimental_distribute_dataset(valid_ds)

    if is_remote:
        epochs = 20
        fit_verbosity = 2
        GCP_BUCKET = "cvs-gcp-csdac-ml-bucket"
        MODEL_PATH = "transformers"
        checkpoint_path = os.path.join("gs://", GCP_BUCKET, MODEL_PATH, "checkpoints", f"save_at_{epochs}")
        tensorboard_path = os.path.join(
            "gs://", GCP_BUCKET, MODEL_PATH, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        model_save_path = os.path.join("gs://", GCP_BUCKET, MODEL_PATH, "model/model.chkpt")
    else:
        train_ds = train_ds.take(10)
        valid_ds = valid_ds.take(1)
        checkpoint_path = os.path.join("checkpoints", f"save_at_{epochs}")
        tensorboard_path = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        model_save_path = os.path.join("model", "0003")

    callbacks = [
        # TensorBoard will store logs for each epoch and graph performance for us.
        tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path, histogram_freq=1),
        # ModelCheckpoint will save models after each epoch for retrieval later.
        # tf.keras.callbacks.ModelCheckpoint(checkpoint_path),
        # EarlyStopping will terminate training when val_loss ceases to improve.
        # tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
    ]

    # optimizer = tf.keras.optimizers.Adam(LRScheduler(d_model, 4000), 0.9, 0.98, 1e-9)
    optimizer = tf.keras.optimizers.Adam()
    model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, heads, d_k, d_v, d_model,
                             d_ff, layers, drop_rate)

    # model.compile(optimizer=optimizer, loss=loss_fcn, metrics=[accuracy_fcn])
    make_a_prediction("---- predictions with empty model", model, valX_enc, valX_dec, valY_dec)
    # hist_obj = model.fit(train_ds, epochs=epochs, validation_data=valid_ds, verbose=fit_verbosity, callbacks=callbacks)
    # model.save_weights(model_save_path)


    # Create a checkpoint object and manager to manage multiple checkpoints
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=None)

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.Mean(name="train_metrics")
    valid_loss = tf.keras.metrics.Mean(name="train_loss")
    valid_accuracy = tf.keras.metrics.Mean(name="train_metrics")
    for epoch in range(epochs):
        epoch_start_time = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()

        for i, ((trainX_enc, trainX_dec), trainY_dec) in enumerate(train_ds):
            loss, accuracy = train_step(model, optimizer, trainX_enc, trainX_dec, trainY_dec)
            train_loss(loss)
            train_accuracy(accuracy)

        for i, ((valX_enc, valX_dec), valY_dec) in enumerate(valid_ds):
            predict = model((valX_enc, valX_dec), training=False)
            loss = loss_fcn(valY_dec, predict)
            accuracy = accuracy_fcn(valY_dec, predict)
            valid_loss(loss)
            valid_accuracy(accuracy)

        print(f"{epoch}/{epochs} ({(time.time() - epoch_start_time):.0f}s): train loss/acc:{train_loss.result():.4f}/{train_accuracy.result():.4f}, valid loss/acc:{valid_loss.result():.4f}/{valid_accuracy.result():.4f}")


        if epoch % 1 == 0:
            # save_path = chkpt_mgr.save()   # TF save mechanism
            # print(f"{epoch=}: save checkpoint {save_path}")
            model.save_weights(f"weights/{epoch:04}.ckpt")  # Keras save mechanism


    make_a_prediction("---- predictions with trained model", model, valX_enc, valX_dec, valY_dec)


    model1 = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, heads, d_k, d_v, d_model,
                             d_ff, layers, rate=0)
    make_a_prediction("---- predictions with empty model1", model1, valX_enc, valX_dec, valY_dec)
    # model1.load_weights(model_save_path)
    model1.load_weights(f"weights/0001.ckpt")
    make_a_prediction("---- predictions with loaded model1", model1, valX_enc, valX_dec, valY_dec)

    return


if __name__ == "__main__":
    main()
