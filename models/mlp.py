import tensorflow as tf

import boilerplate as tfbp


@tfbp.default_export
class MLP(tfbp.Model):
    hyperparameters = {"batch_size"}

    def call(self):
        pass

    def train_loop(self):
        pass
