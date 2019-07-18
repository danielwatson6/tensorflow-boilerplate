# TensorFlow Boilerplate

This repository contains a simple workflow to work efficiently with TensorFlow 2.0. It removes the need to write training scripts for each new model, as well as code gluing models and input pipelines together.

## Setup

**No dependencies needed besides Python 3.7.4, virtualenv TensorFlow 2.0.0-beta1.** Start developing your new model on top of this workflow by cloning this repository:

```bash
git clone https://github.com/danielwatson6/tensorflow-boilerplate.git
cd tensorflow-boilerplate
virtualenv env
source env.sh
pip install tensorflow==2.0.0-beta1  # or tensorflow-gpu==2.0.0-beta1 / custom wheel
```

## Usage

You should run `source env.sh` on each new shell session. This activates the virtualenv and creates a nice alias for `run.py`:
```bash
$ cat env.sh
source env/bin/activate
alias run='python run.py'
```

The repository contains a few directories:
- `data`: gitignore'd directory to place data
- `experiments`: trained models written here
- `data_loaders`: write your data loaders here
- `models`: write your models here

Most routines involve running a command like this:
```bash
# Usage: run [method] [experiment_slug] [model] [data_loader] [hparams...]
run fit myexperiment1 gan mnist --batch_size=32 --learning_rate=0.1
```

where the `model` and `data_loader` args are the module names (i.e., the file names without the `.py`). The command above would run the Keras model's `fit` method, but it could be any custom as long as it accepts a data loader instance as argument.

**If `save_dir` already has a model, only the first two arguments are required, but the data loader may be changed.**

Each of these modules live in their respective directories and should export a default subclass of either of the following:

```python
tfbp.Model   # Extends `tf.keras.Model`
tfbp.DataLoader
```

Both of these classes have minimal functionality to allow command like parsing like shown above. Since that includes parsing hyperparameters, they can be defined statically with default values by setting `default_hparams`:

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

After the constructor is called, the model can access these hyperparameters, e.g., `self.hparams.learning_rate`, `self.hparams.hidden_size`, etc. `self.hparams` is a namedtuple.

The method string argument received by `run.py` is also accessible in both classes as `self.method`, which can be useful to separate cases between training, inference, and other routines as needed.


### `tfbp.Model`

Models pretty much follow the same rules as Keras models with very slight differences: the constructor's arguments should not be overriden (since the boilerplate code handles instantiation), and new `save` and `restore` methods are available.

```python
import tensorflow as tf
import boilerplate as tfbp

@tfbp.default_export
class MyModel(tfbp.Model):
    default_hparams = {
        "batch_size": 32,
        "hidden_size": 512,
        "learning_rate": 0.01,
    }

    # Don't mess with the args and keyword args, `run.py` handles that.
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

        self.dense1 = tf.keras.layers.Dense(self.hparams.hidden_size)
        ...

    def call(self, x):
        z = self.dense1(x)
        ...
```

You can also write your own training loops à la pytorch by overriding the `fit` method
or writing a custom method that you can invoke via `run.py`.

### `tfbp.DataLoader`

Since model methods invoked by `run.py` receive a data loader instance, you may name your data loader whatever you wish and call them in your model code. A good practice is to make the data loader handle anything that is specific to a particular dataset, allowing the model to be as general as possible.

```python
import tensorflow as tf
import boilerplate as tfbp

@tfbp.default_export
class MyDataLoader(tfbp.DataLoader):
    default_hparams = {
        "batch_size": 32,
    }

    def load(self):
        if self.method == "fit":
            train_data = tf.data.TextLineDataset("data/train.txt")
            valid_data = tf.data.TextLineDataset("data/valid.txt")
            dataset = tf.data.Dataset.zip((train_data, valid_data)).shuffle(10000)

        elif self.method == "eval":
            dataset = tf.data.TextLineDataset("data/test.txt")

        return dataset.batch(self.hparams.batch_size).prefetch(1)
```
