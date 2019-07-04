import tensorflow as tf
from tensorflow.keras import layers as tfkl

import boilerplate as tfbp


@tfbp.default_export
class MLP(tfbp.Model):
    default_hparams = {"layer_sizes": [100, 10], "learning_rate": 0.1}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward = tf.keras.Sequential()
        for hidden_size in self.hparams.layer_sizes[:-1]:
            self.forward.add(tfkl.Dense(hidden_size, activation=tf.nn.relu))
        self.forward.add(
            tfkl.Dense(self.hparams.layer_sizes[-1], activation=tf.nn.softmax)
        )

    def call(self, x):
        if self.training:
            x, y = x

        x = self.forward(x)
        outputs = tf.argmax(x, axis=-1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(outputs, y), tf.float32))

        if not self.training:
            return outputs, accuracy

        loss = tf.losses.CategoricalCrossentropy(labels=y, logits=x)
        return loss, outputs, accuracy

    def fit(self, dataset):
        step = 0
        train_data, _ = dataset

        optimizer = tf.optimizers.SGD(self.hparams.learning_rate)

        for batch in train_data:
            if not self.built:
                self.build([tf.shape(batch[0]), tf.shape(batch[1])])

            with tf.GradientTape() as g:
                loss, outputs, accuracy = self(batch)
            grads = g.gradients(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.trainable_variables))

            print(f"step {step} (train_loss={loss})")

            if self.step % 1000 == 0:
                tf.summary.scalar("loss", loss)
                tf.summary.scalar("accuracy", accuracy)
                self.save()
        self.save()
