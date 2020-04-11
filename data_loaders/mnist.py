import tensorflow as tf

import boilerplate as tfbp


@tfbp.default_export
class MNIST(tfbp.DataLoader):
    @staticmethod
    def hparams(hp):
        # https://keras-team.github.io/keras-tuner/documentation/hyperparameters/
        hp.Int("batch_size", 16, 512, default=32, sampling="log")

    def __call__(self):
        train_data, test_data = tf.keras.datasets.mnist.load_data()
        test_data = tf.data.Dataset.from_tensor_slices(test_data)
        test_data = self._transform_dataset(test_data)

        if self.method in ["fit", "train"]:
            train_data = tf.data.Dataset.from_tensor_slices(train_data).shuffle(10000)
            test_data = test_data.shuffle(10000)
            train_data = self._transform_dataset(train_data)
            return train_data, test_data

        return test_data

    def _transform_dataset(self, dataset):
        dataset = dataset.batch(self.hp.batch_size)
        return dataset.map(
            lambda x, y: (
                tf.reshape(tf.cast(x, tf.float32) / 255.0, [-1, 28 * 28]),
                tf.cast(y, tf.int64),
            )
        )
