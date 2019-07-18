import tensorflow as tf

import boilerplate as tfbp


@tfbp.default_export
class MNIST(tfbp.DataLoader):
    default_hparams = {"batch_size": 32}

    def load(self):
        train_data, test_data = tf.keras.datasets.mnist.load_data()

        test_data = tf.data.Dataset.from_tensor_slices(test_data)
        test_data = self._transform_dataset(test_data)

        if self.method in ["fit", "train"]:
            train_data = tf.data.Dataset.from_tensor_slices(train_data)
            train_data = self._transform_dataset(train_data)
            return train_data, test_data

        return test_data

    def _transform_dataset(self, dataset):
        dataset = dataset.shuffle(10000)
        dataset = dataset.batch(self.hparams.batch_size)
        return dataset.map(
            lambda x, y: (
                tf.reshape(tf.cast(x, tf.float32), [-1, 28 * 28]),
                tf.cast(y, tf.int64),
            )
        )
