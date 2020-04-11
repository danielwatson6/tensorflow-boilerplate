"""TensorFlow Boilerplate main module."""

from collections import namedtuple
from copy import deepcopy
from functools import wraps
import json
import os
import sys

import tensorflow as tf
import kerastuner as kt


class Model(tf.keras.Model):
    """Keras model with hyperparameter parsing and a few other utilities."""

    # Dictionary keeping track of all the runnable methods (i.e., those that can be
    # invoked via the CLI). This gets populated in a `tfbp.Model` subclass by decorating
    # it with `tfbp.default_export`.
    _runnables = {}

    def __init__(self, save_dir=None, method=None, hparams=None):
        """Constructor.

        Kwargs:
            save_dir: model files will be saved in experiments/[save_dir]
            method: model's current runnable
            hparams: the model's hyperparameters.
        """
        super().__init__()
        self._save_dir = save_dir
        self._method = method
        self.hp = hparams

        self._ckpt = None
        self._ckpt_manager = None

    @property
    def save_dir(self):
        """Get the directory where the model files are saved."""
        return self._save_dir

    @property
    def method(self):
        """Get the model's current runnable."""
        return self._method

    @staticmethod
    def hparams(hp):
        """Get the model's hyperparameters.

        This should be overriden by any model with hyperparameters, which are specified
        simply by making calls to `hp` that create or retrieve hyperparameters. Example:

        @staticmethod
        def hparams(hp):
            hp.Int("batch_size", 16, 1024, default=32, sampling="log")
            hp.Float("learning_rate")

        https://keras-team.github.io/keras-tuner/documentation/hyperparameters/
        """

    @property
    def hp(self):
        return self._hp

    @hp.setter
    def hp(self, hparams):
        if isinstance(hparams, kt.HyperParameters):
            kt_hparams = hparams
        elif type(hparams) == dict:
            # Override the default hyperparameters with the given values.
            kt_hparams = kt.HyperParameters()
            for name, value in hparams.items():
                kt_hparams.Fixed(name, value)
        else:
            raise ValueError(f"Hyperparameter type not supported: {type(hparams)}")

        # Fill in the rest of the hyperparameters that were not specified.
        self.hparams(kt_hparams)

        # Make the hyperparameters accessible as a namedtuple for convenience.
        _HP = namedtuple("HyperParameters", kt_hparams.values.keys())
        self._hp = _HP(**kt_hparams.values)

    def _build_ckpt(self):
        self._ckpt = tf.train.Checkpoint(model=self)
        self._ckpt_manager = tf.train.CheckpointManager(
            self._ckpt, directory=self.save_dir, max_to_keep=1
        )

    def save(self):
        """Save the model's weights and hyperparameters."""
        if self._ckpt is None:
            self._build_ckpt()
        self._ckpt_manager.save()
        with open(os.path.join(self.save_dir, "hparams.json"), "w") as f:
            json.dump(self.hp._asdict(), f, indent=4, sort_keys=True)

    # TODO: get rid of `hp` and load the model's hparams.json.
    @classmethod
    def load(cls, save_dir, method=None, hparams=None):
        """Load a saved model from the given directory."""
        if type(hparams) == dict:
            with open(os.path.join(self.save_dir, "hparams.json")) as f:
                hparams = {**json.load(f), **hparams}

        model = cls(save_dir=save_dir, method=method, hparams=hparams)
        if not os.path.isfile(os.path.join(save_dir, "checkpoint")):
            raise FileNotFoundError(f"No model saved at `{save_dir}`.")
        model.restore()
        return model

    def restore(self):
        """Restore the model's latest saved weights."""
        if self._ckpt is None:
            self._build_ckpt()
        self._ckpt.restore(self._ckpt_manager.latest_checkpoint)

    def make_summary_writer(self, dirname):
        """Create a TensorBoard summary writer."""
        return tf.summary.create_file_writer(os.path.join(self.save_dir, dirname))

    @classmethod
    def get_runnable(cls, method):
        if method not in cls._runnables:
            methods_str = "\n  ".join(cls._runnables.keys())
            raise ValueError(
                f"`{cls.__name__}` does not have a runnable method `{FLAGS.method}`. "
                f"Methods available:\n  {methods_str}"
            )
        return cls._runnables.get(method)

    def run(self, data_loader):
        """Call the model's runnable function as specified by its `method` attribute."""
        run_fn = self.get_runnable(self.method)
        return run_fn(self, data_loader)


class DataLoader:
    """Data loader class akin to `Model`."""

    def __init__(self, method=None, hparams=None):
        self._method = method
        self.hp = hparams

    @property
    def method(self):
        """Get the model's current runnable."""
        return self._method

    @staticmethod
    def hparams(hp):
        """Get the model's hyperparameters.

        This should be overriden by any model with hyperparameters, which are specified
        simply by making calls to `hp` that create or retrieve hyperparameters. Example:

        @staticmethod
        def hparams(hp):
            hp.Int("batch_size", 16, 1024, default=32, sampling="log")
            hp.Float("learning_rate")

        https://keras-team.github.io/keras-tuner/documentation/hyperparameters/
        """

    @property
    def hp(self):
        return self._hp

    @hp.setter
    def hp(self, hparams):
        if isinstance(hparams, kt.HyperParameters):
            kt_hparams = hparams
        elif type(hparams) == dict:
            # Override the default hyperparameters with the given values.
            kt_hparams = kt.HyperParameters()
            for name, value in hparams.items():
                kt_hparams.Fixed(name, value)
        else:
            raise ValueError(f"Hyperparameter type not supported: {type(hparams)}")

        # Fill in the rest of the hyperparameters that were not specified.
        self.hparams(kt_hparams)

        # Make the hyperparameters accessible as a namedtuple for convenience.
        _HP = namedtuple("HyperParameters", kt_hparams.values.keys())
        self._hp = _HP(**kt_hparams.values)


class _HyperModel(kt.engine.hypermodel.HyperModel):
    def __init__(self, Model, **kw):
        super().__init__(**kw)
        self._model_cls = Model

    def build(self, hparams, **kw):
        return self._model_cls(hparams=hparams, **kw)


class Tuner(kt.engine.base_tuner.BaseTuner):
    def __init__(
        self, oracle, Model_, save_dir=None, train_method=None, eval_method=None, **kw,
    ):
        """Constructor.

        Args
            oracle: a `kerastuner.Oracle` instance
            Model: a `tfbp.Model` subclass

        Keyword args
            save_dir: directory to save all metadata files
            train_method: the name of the method used to train the model
            eval_method: a list/tuple. Must be a pair consisting of the name of the
                method used to score the model, along with either "min" or "max" to know
                whether higher or lower values correspond to a better model
        """
        super().__init__(
            oracle,
            _HyperModel(Model_),
            directory=save_dir,
            project_name=".kerastuner",
            **kw,
        )
        self._train_method = train_method
        self._eval_method = eval_method

    def _populate_initial_space(self):
        return

    def search(self, DataLoader_):
        """Perform a hyperparameter search.

        `BaseTuner` passess all the args and kwargs to `run_trial`.

        Args
            DataLoader: a `tfbp.DataLoader` subclass
        """
        return super().search(DataLoader_)

    def run_trial(self, trial, DataLoader_):
        save_dir = os.path.join(self.directory, trial.trial_id)
        data_loader = DataLoader_(hparams=trial.hyperparameters)

        train_model = self.hypermodel.build(
            trial.hyperparameters, method=self._train_method, save_dir=save_dir
        )
        train_model.run(data_loader)
        train_model.save()

        eval_model = self.hypermodel.build(
            trial.hyperparameters, method=self._eval_method, save_dir=save_dir
        )
        eval_model.restore()
        metrics = {}
        metrics[self._eval_method] = eval_model.run(data_loader)
        print(metrics)
        self.oracle.update_trial(trial.trial_id, metrics)


def runnable(f):
    """Mark a method as runnable from `run.py`."""

    @wraps(f)
    def wrapped(self, data_loader):
        method = f.__name__
        self._method = method
        data_loader._method = method
        return f(self, data_loader)

    setattr(wrapped, "_runnable", True)
    return wrapped


def default_export(cls):
    """Make the class the imported object of the module and collect its runnables."""
    newcls = deepcopy(cls)

    for name, method in vars(newcls).items():
        if "_runnable" in dir(method):
            newcls._runnables[name] = method

    sys.modules[cls.__module__] = newcls
    return newcls


def get_model(module_str):
    """Import models.[module_str]."""
    return getattr(__import__(f"models.{module_str}"), module_str)


def get_data_loader(module_str):
    """Import data_loaders.[module_str]."""
    return getattr(__import__(f"data_loaders.{module_str}"), module_str)
