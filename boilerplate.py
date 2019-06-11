from argparse import ArgumentParser
from collections import namedtuple

import tensorflow as tf


def Hyperparameters(value):
    # Don't transform the value if it's a namedtuple.
    # https://stackoverflow.com/questions/2166818/how-to-check-if-an-object-is-an-instance-of-a-namedtuple
    t = type(value)
    b = t.__bases__
    if len(b) == 1 and b[0] == tuple:
        f = getattr(t, "_fields", None)
        if isinstance(f, tuple) and all(type(n) == str for n in f):
            return value

    _Hyperparameters = namedtuple("Hyperparameters", value.keys())
    return _Hyperparameters(**value)


class Model(tf.keras.Model):
    hparams = {}

    def __init__(self):
        self.hparams = Hyperparameters(self.hparams)

    @hparams.setter
    def hparams(self, value):
        self._hparams = Hyperparameters(value)

    def get_parser(self):
        parser = ArgumentParser()
        parser.add_argument("model", type=str)
        parser.add_argument("save_dir", type=str)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--epochs", type=int, default=32)
        for name, default_value in self.hparams._asdict().items():
            parser.add_argument(f"--{name}", type=type(value), default=value)
        return parser.parse_args()


def default_export(cls):
    """Decorator to make a class or method the imported object of a module."""
    sys.modules[cls.__module__] = cls
    return cls
