# TensorFlow Boilerplate

This repository contains a simple workflow to work efficiently with TensorFlow 2.0. It removes the need to write training scripts for each new model, as well as code gluing models and input pipelines together.

## Setup

*No dependencies needed besides Python 3.7.3 and TensorFlow 2.0.0b0.* Start developing your new model on top of this workflow by cloning this repository:

```bash
git clone https://github.com/danielwatson6/tensorflow-boilerplate.git
```

## Usage

The repository contains a few directories:
- `data`: gitignore'd directory to place data
- `experiments`: trained models written here
- `data_loaders`: write your data loaders here
- `models`: write your models here

The `boilerplate` module contains two classes that are meant to be subclassed:

```python


tfbp.Model   # Extends `tf.keras.Model`
tfbp.DataLoader
```

By writing the subclasses as individual modules in the `models` or `data_loaders` directories, they can be glued together and run with `run.py` without the need to do anything else. This also allows to specify the hyperparameters needed which `run.py` can then read in the command line:

```python
import boilerplate as tfbp

# When importing this module, the read module object will be this class.
@tfbp.default_export
class MyModel(tfbp.Model):  # or `tfbp.DataLoader`

    # Set the `default_hparams` static variable, listing every required hyperparameter.
    default_hparams = {
        "learning_rate": 0.01,
        "hidden_size": 512,
    }
```


### `tfbp.Model`

Models pretty much follow the same rules as Keras models.

```python
import tensorflow as tf
import boilerplate as tfbp

@tfbp.default_export
class MyModel(tfbp.Model):
    default_hparams = {
        "batch_size": 32,
        "hidden_size": 512,
        "learning_rate": 0.0.1,
    }

    # Don't mess with the args and keyword args, `run.py` handles that.
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

        # As usual, all the layers are defined below to be used in `call`.
        #
        # NOTE: the following new methods and properties are now usable:
        #
        # Properties:
        #   save_dir (str): full output path, e.g. "experiments/mymodel_run1".
        #   method (str): the method currently running, e.g., "fit" or "evaluate".
        #   hparams (namedtuple): the actual (not necessarily default) hyperparameters.
        #
        # Methods:
        #   save: takes no arguments, saves the model
        #   restore: takes no arguments, restores the model

        self.dense1 = tf.keras.layers.Dense(self.hparams.hidden_size)
        ...

    def call(self, x):
        z = self.dense1(x)
        ...
```

You can also write your own training loops à la pytorch by overriding the `fit` method.

### `tfbp.DataLoader`

Same story as models, but only the `load` method needs to be written:

```python
import tensorflow as tf
import boilerplate as tfbp

@tfbp.default_export
class MyDataLoader(tfbp.DataLoader):
    default_hparams = {
        "batch_size": 32,
    }

    def load(self):
        # Whatever is returned here will be passed to one of the model's methods like
        # `fit` or `evaluate`, whichever is specified in `run.py`.
        #
        # A good practice is to return a `tf.data.Dataset` instance, or separate
        # training and validation dataset instances for training. You have access to
        # `self.method` and `self.hparams`, just like in the `tfbp.Model` class.
        ...

### `run.py`

Run any of your model's bound methods (e.g. `fit` or `evaluate`) as follows:
```bash
# Args:
#   method: model's bound method to call (passing the data loader's output)
#   save_dir: directory name to save or restore the model
#   model: filename (minus .py) of the model
#   data_loader: filename (minus .py) of the data loader
#   hparams: any hyperparameters in the format --name==value

python run.py [method] [model] [save_dir] [model] [data_loader] [hparams...]
```

*If `save_dir` already has a model, only the first two arguments are required.*
