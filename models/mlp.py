import tensorflow as tf
from tensorflow.keras import layers as tfkl

import boilerplate as tfbp


@tfbp.default_export
class MLP(tfbp.Model):
    """Example implementation of a multilayer perceptron."""

    @staticmethod
    def hparams(hp):
        # https://keras-team.github.io/keras-tuner/documentation/hyperparameters/
        hp.Fixed("num_classes", 10)
        hp.Int("num_hidden", 1, 3, default=1)
        hp.Int("hidden_size", 128, 512, default=256, sampling="log")
        hp.Choice("activation", ["tanh", "relu"], default="tanh")
        hp.Float("learning_rate", 5e-4, 5e-3, default=1e-3, sampling="log")
        hp.Float("dropout", 0.1, 0.4, default=0.1)
        hp.Fixed("epochs", 3)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.forward = tf.keras.Sequential()

        for _ in range(self.hp.num_hidden):
            self.forward.add(
                tfkl.Dense(self.hp.hidden_size, activation=self.hp.activation)
            )
            self.forward.add(tfkl.Dropout(self.hp.dropout))
        self.forward.add(tfkl.Dense(self.hp.num_classes, activation=tf.math.sigmoid))

        self.loss = tf.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.optimizers.Adam(self.hp.learning_rate)

    # This is necessary when using the `compile` functionality of Keras.
    def call(self, x):
        return self.forward(x)

    @tfbp.runnable
    def fit(self, data_loader):
        """Example using keras training loop."""
        train_data, valid_data = data_loader()

        self.compile(self.optimizer, self.loss)
        super().fit(
            x=train_data,
            validation_data=valid_data,
            validation_steps=32,  # validate 32 batches at a time
            validation_freq=1,  # validate every 1 epoch
            epochs=self.hp.epochs,
            shuffle=False,  # dataset instances already handle shuffling
        )
        self.save()

    @tfbp.runnable
    def train(self, data_loader):
        """Example using custom training loop."""
        train_data, valid_data = data_loader()

        # Allow to call `next` builtin indefinitely.
        valid_data = iter(valid_data.repeat())

        # This is in its own function just for the sake of performance speedup. By
        # decorating `train_fn` with `tf.function`, the train loop runs way faster.
        self.train_fn(train_data, valid_data)
        self.save()

    @tf.function
    def train_fn(self, train_data, valid_data):
        step = 0
        for epoch in range(self.hp.epochs):
            for x, y in train_data:

                with tf.GradientTape() as g:
                    train_loss = self.loss(y, self(x))

                grads = g.gradient(train_loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

                # Validate every 1000 training steps.
                if step % 1000 == 0:
                    x, y = next(valid_data)
                    valid_loss = self.loss(y, self(x))
                    tf.print("step", step)
                    tf.print("  train_loss:", train_loss)
                    tf.print("  valid_loss:", valid_loss)
                step += 1

            tf.print("epoch", epoch, "finished")

    @tfbp.runnable
    def accuracy(self, data_loader):
        test_data = data_loader()

        # Running average.
        total_acc = 0.0
        i = 0.0
        for x, y in test_data:
            batch_acc = tf.math.equal(y, tf.math.argmax(self(x), axis=-1))
            batch_acc = tf.cast(batch_acc, tf.float32)
            for acc in batch_acc:
                i += 1
                total_acc += (acc - total_acc) / i

        tf.print(total_acc)
        return total_acc
