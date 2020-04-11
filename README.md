# TensorFlow Boilerplate (v2.0)

This repository contains a simple workflow to work efficiently with TensorFlow 2.0. It removes the need to write training scripts for each new model, code gluing models and input pipelines together, and automatically tuning hyperparameters.


## Usage

This boilerplate provides two abstractions: **models** and **data loaders**. Typically, models are Keras models that support training and evaluations, and data loaders arethe input pipelines. The core principle is to develop your models so they are as independent as possible to the data fed to them.

Most routines involve running a command like this:
```bash
# Usage: run [method] [save_dir] [model] [data_loader] [hparams...]
run fit myexperiment1 mlp mnist --batch_size=32 --learning_rate=0.1
```

where the `model` and `data_loader` args are the module names (i.e., the file names without the `.py`). The command above would run the Keras model's `fit` method, but this can be switched to any other method flagged as runnable in the model.

**Check `models/mlp.py` and `data_loaders/mnist.py` for fully working examples.**

### Directory structure

- `data`: gitignore'd, place datasets here.
- `experiments`: gitignore'd, model files are written here.
- `data_loaders`: write your data loaders here.
- `models`: write your models here.


## Setup

**Make sure to have the lastest python 3.7 version installed.**

Start developing your new model on top of this workflow by cloning this repository:

```bash
git clone https://github.com/danielwatson6/tensorflow-boilerplate.git
cd tensorflow-boilerplate

# Recommended: set up a virtualenv
virtualenv env
source env.sh

pip install -r requirements.txt
```

### (Optional) receive updates from tensorflow-boilerplate

Add a `tfbp` pull-only remote pointing to tensorflow-boilerplate, just once:

```bash
git remote add tfbp https://github.com/danielwatson6/tensorflow-boilerplate
git remote set-url --push tfbp -- --read-only--
```

Update the boilerplate via `git pull tfbp master` as often as needed.
