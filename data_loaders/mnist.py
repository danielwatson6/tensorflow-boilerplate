import tensorflow as tf

import boilerplate as tfbp


@tfbp.default_export
class MNIST(tfbp.DataLoader):
    default_hparams = {"batch_size": 32}

    def load(self):
        train_data, test_data = tf.keras.datasets.mnist.load_data()

        test_data = tf.data.Dataset.from_tensor_slices(test_data)
        test_data = test_data.batch(self.hparams.batch_size)

        if self.training:
            train_data = tf.data.Dataset.from_tensor_slices(train_data)
            train_data = train_data.batch(self.hparams.batch_size)
            return train_data, test_data

        return test_data
