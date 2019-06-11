import sys

import tensorflow as tf

import boilerplate


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Usage: python train.py [save_dir] [model] [data_loader] "
            "[hyperparameters...]",
            file=sys.stderr,
        )
        exit(1)

    # Initialize new model.
    if len(sys.argv) >= 4 and not sys.argv[3].startswith("--"):
        Model = __import__("models." + sys.argv[2])
        Dataset = __import__("models." + sys.argv[3])

    # Restore previously existing model:
    else:

