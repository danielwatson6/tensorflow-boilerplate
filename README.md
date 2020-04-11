# TensorFlow Boilerplate (v1.0)

This repository contains a simple workflow to work efficiently with TensorFlow 2.0. It removes the need to write training scripts for each new model, as well as code gluing models and input pipelines together.

## Setup

**No dependencies needed besides Python 3.7.4, virtualenv, and TensorFlow.** Start developing your new model on top of this workflow by cloning this repository:

```bash
git clone https://github.com/danielwatson6/tensorflow-boilerplate.git
cd tensorflow-boilerplate

# Recommended: set up a virtualenv
virtualenv env
source env.sh

pip install tensorflow
```

### (Optional) receive updates from tensorflow-boilerplate

Add a `boilerplate` pull-only remote pointing to tensorflow-boilerplate, just once:

```bash
git remote add tfbp https://github.com/danielwatson6/tensorflow-boilerplate
git remote set-url --push boilerplate -- --read-only--
```

Update the boilerplate via `git pull tfbp master` as often as needed.


## Directory structure

- `data`: gitignore'd, place datasets here.
- `experiments`: gitignore'd, trained models written here.
- `data_loaders`: write your data loaders here.
- `models`: write your models here.


## Usage

**Check `models/mlp.py` and `data_loaders/mnist.py` for fully working examples.**

You should run `source env.sh` on each new shell session. This activates the virtualenv and creates a nice alias for `run.py`:
```bash
$ cat env.sh
source env/bin/activate
alias run='python run.py'
```

Most routines involve running a command like this:
```bash
# Usage: run [method] [save_dir] [model] [data_loader] [hparams...]
run fit myexperiment1 mlp mnist --batch_size=32 --learning_rate=0.1
```

where the `model` and `data_loader` args are the module names (i.e., the file names without the `.py`). The command above would run the Keras model's `fit` method, but it could be any custom as long as it accepts a data loader instance as argument.

**If `save_dir` already has a model**:
- Only the first two arguments are required and the data loader may be changed, but respecifying the model is not allowed-- the existing model will always be used.
- Specified hyperparameter values in the command line WILL override previously used ones
(for this run only, not on disk).


### `tfbp.Model`

Models pretty much follow the same rules as Keras models with very slight differences: the constructor's arguments should not be overriden (since the boilerplate code handles instantiation), and the `save` and `restore` methods don't need any arguments.

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
or writing a custom method that you can invoke via `run.py` simply by adding the
`@tfbp.runnable` decorator. Examples of both are available in `models/mlp.py`.

### `tfbp.DataLoader`

Since model methods invoked by `run.py` receive a data loader instance, you may name your data loader methods whatever you wish and call them in your model code. A good practice is to make the data loader handle anything that is specific to a particular dataset, which allows the model to be as general as possible.

```python
import tensorflow as tf
import boilerplate as tfbp

@tfbp.default_export
class MyDataLoader(tfbp.DataLoader):
    default_hparams = {
        "batch_size": 32,
    }

    def __call__(self):
        if self.method == "fit":
            train_data = tf.data.TextLineDataset("data/train.txt").shuffle(10000)
            valid_data = tf.data.TextLineDataset("data/valid.txt").shuffle(10000)
            return self.prep_dataset(train_data), self.prep_dataset(valid_data)

        elif self.method == "eval":
            test_data = tf.data.TextLineDataset("data/test.txt")
            return self.prep_dataset(test_data)

    def prep_dataset(self, ds):
        return ds.batch(self.hparams.batch_size).prefetch(1)
```
