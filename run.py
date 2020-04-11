"""Script to execute any runnable in a TensorFlow Boilerplate model."""

from argparse import ArgumentParser
import json
import os
import sys

from kerastuner import HyperParameters

import boilerplate as tfbp


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage:\n  New run: python run.py [method] [save_dir] [model] [data_loader]",
            "[hyperparameters...]\n  Existing run: python run.py [method] [save_dir]",
            "[data_loader]? [hyperparameters...]",
            file=sys.stderr,
        )
        exit(1)

    save_dir = os.path.join("experiments", sys.argv[2])
    # Avoid errors due to a missing directories.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Dynamically parse arguments from the command line depending on the model and data
    # loader provided. The `method` and `save_dir` arguments are always required.
    parser = ArgumentParser()
    parser.add_argument("method", type=str)
    parser.add_argument("save_dir", type=str)

    # If run_metadata.json exists, the model and the data loader modules can be inferred,
    # and the data loader can be optionally changed from its default.
    run_metadata_path = os.path.join(save_dir, "run_metadata.json")
    if os.path.isfile(run_metadata_path):

        with open(run_metadata_path) as f:
            classes = json.load(f)

        if len(sys.argv) >= 4 and not sys.argv[3].startswith("--"):
            classes["data_loader"] = sys.argv[3]

        Model = tfbp.get_model(classes.get("model"))

        # The model shouldn't be provided for an existing run, but for convenience this
        # error is handled for the user.
        try:
            DataLoader = tfbp.get_data_loader(classes.get("data_loader"))
            parser.add_argument("data_loader", type=str)
        except ModuleNotFoundError:
            if len(sys.argv) < 5 or sys.argv[4].startswith("--"):
                raise

            print(
                f"Warning: model already specified at {save_dir}",
                "ignoring some args...",
                file=sys.stderr,
            )
            classes["data_loader"] = sys.argv[4]
            DataLoader = tfbp.get_data_loader(classes.get("data_loader"))
            parser.add_argument("model", type=str)
            parser.add_argument("data_loader", type=str)

    else:
        Model = tfbp.get_model(sys.argv[3])
        DataLoader = tfbp.get_data_loader(sys.argv[4])

        parser.add_argument("model", type=str)
        parser.add_argument("data_loader", type=str)

        with open(run_metadata_path, "w") as f:
            json.dump(
                {"model": sys.argv[3], "data_loader": sys.argv[4]},
                f,
                indent=4,
                sort_keys=True,
            )

    khp = HyperParameters()
    # Load both the model and the data loader's hyperparameters into `hparams`.
    Model.hparams(khp)
    DataLoader.hparams(khp)

    for p in khp.space:
        # TODO: add support for falsy values for boolean hyperparameters.
        # TODO: restrict hyperparameters to their allowed values.
        parser.add_argument(f"--{p.name}", type=type(p.default), default=p.default)

    # Collect parsed hyperparameters.
    hparams = {
        k: v
        for k, v in parser.parse_args()._get_kwargs()
        if k not in {"method", "save_dir", "model", "data_loader"}
    }

    model = Model(save_dir=save_dir, method=sys.argv[1], hparams=hparams)
    data_loader = DataLoader(hparams=hparams)
    model.run(data_loader)
