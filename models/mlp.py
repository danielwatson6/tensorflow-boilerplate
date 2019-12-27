import tensorflow as tf
from tensorflow.keras import layers as tfkl

import boilerplate as tfbp


@tfbp.default_export
class MLP(tfbp.Model):
    default_hparams = {
        "layer_sizes": [512, 10],
        "learning_rate": 0.001,
        "num_epochs": 10,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.forward = tf.keras.Sequential()

        for hidden_size in self.hparams.layer_sizes[:-1]:
            self.forward.add(tfkl.Dense(hidden_size, activation=tf.nn.relu))

        self.forward.add(
            tfkl.Dense(self.hparams.layer_sizes[-1], activation=tf.nn.softmax)
        )

    def call(self, x):
        return self.forward(x)

    @tfbp.runnable
    def fit(self, data_loader):
        """Example using keras training loop."""
        train_data, valid_data = data_loader.load()

        loss_fn = tf.losses.SparseCategoricalCrossentropy()
        optimizer = tf.optimizers.Adam(self.hparams.learning_rate)

        self.compile(optimizer, loss_fn)
        super().fit(
            x=train_data,
            validation_data=valid_data,
            validation_steps=1,
            validation_freq=1,
            epochs=self.hparams.num_epochs,
            shuffle=False,
        )
        self.save()

    @tfbp.runnable
    def train(self, data_loader):
        """Example using custom training loop."""
        step = 0
        train_data, valid_data = data_loader.load()

        # Allow to call `next` builtin indefinitely.
        valid_data = iter(valid_data.repeat())

        loss_fn = tf.losses.SparseCategoricalCrossentropy()
        optimizer = tf.optimizers.Adam(self.hparams.learning_rate)

        for epoch in range(self.hparams.num_epochs):
            for x, y in train_data:

                with tf.GradientTape() as g:
                    loss = loss_fn(y, self(x))

                grads = g.gradient(loss, self.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.trainable_variables))

                if step % 1000 == 0:
                    print(f"step {step} (train_loss={loss})")
                step += 1

            # Save and validate at the end of every epoch.
            x, y = next(valid_data)
            loss = loss_fn(y, self(x))
            print(f"epoch {epoch} finished (valid_loss={loss})")
            self.save()

    @tfbp.runnable
    def evaluate(self, data_loader):
        n = 0
        accuracy = 0
        test_data = data_loader.load()
        for x, y in test_data:
            true_pos = tf.math.equal(y, tf.math.argmax(self(x), axis=-1))
            for i in true_pos.numpy():
                n += 1
                accuracy += (i - accuracy) / n
        print(accuracy)
