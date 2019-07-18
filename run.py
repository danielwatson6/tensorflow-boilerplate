from argparse import ArgumentParser
import json
import os
import sys

import tensorflow as tf

import boilerplate


def getcls(module_str):
    head, tail = module_str.split(".")
    return getattr(__import__(f"{head}.{tail}"), tail)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(
            "Usage:\n  New run: python run.py [method] [save_dir] [model] [data_loader]"
            " [hyperparameters...]\n  Existing run: python run.py [method] [save_dir] "
            "[data_loader]? [hyperparameters...]",
            file=sys.stderr,
        )
        exit(1)

    # TODO: make it possible to infer the model and data_loader fields by reading the
    # runpy.json file.

    parser = ArgumentParser()
    parser.add_argument("method", type=str)
    parser.add_argument("save_dir", type=str)

    # If runpy.json exists, the model and the data loader classes can be inferred and
    # the data loader can be optionally switched. These need to be loaded to get the
    # static default hyperparameters to be read by argparse.
    runpy_json_path = os.path.join("experiments", sys.argv[2], "runpy.json")
    if os.path.exists(runpy_json_path):

        with open(runpy_json_path) as f:
            classes = json.load(f)

        if len(sys.argv) >= 4 and not sys.argv[4].startswith("--"):
            classes["data_loader"] = sys.argv[4]

        Model = getcls("models." + classes["model"])
        DataLoader = getcls("data_loaders." + classes["data_loader"])

    else:
        Model = getcls("models." + sys.argv[3])
        DataLoader = getcls("data_loaders." + sys.argv[4])

        parser.add_argument("model", type=str)
        parser.add_argument("data_loader", type=str)

        with open(runpy_json_path, "w") as f:
            json.dump({"model": sys.argv[3], "data_loader": sys.argv[4]}, f)

    args = {}
    for name, value in Model.default_hparams.items():
        args[name] = value
    for name, value in DataLoader.default_hparams.items():
        args[name] = value

    for name, value in args.items():
        parser.add_argument(f"--{name}", type=type(value), default=value)

    FLAGS = parser.parse_args()
    kwargs = {k: v for k, v in FLAGS._get_kwargs()}

    del kwargs["model"]
    del kwargs["save_dir"]
    del kwargs["data_loader"]

    model = Model(os.path.join("experiments", FLAGS.save_dir), **kwargs)
    data_loader = DataLoader(**kwargs)

    try:
        model.restore()
    except Exception:
        model.save()
        with open(os.path.join("experiments", FLAGS.save_dir, "runpy.json"), "w") as f:
            json.dump({"model": FLAGS.model, "data_loader": FLAGS.data_loader}, f)

    getattr(model, FLAGS.method)(data_loader)
