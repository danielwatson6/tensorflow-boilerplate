"""Script to automatically tune a TensorFlow Boilerplate model's hyperparameters."""

from argparse import ArgumentParser
import json
import os
import re
import sys

import kerastuner as kt

import boilerplate as tfbp


def search_hyperparameters(save_dir, config):
    Model = tfbp.get_model(config.get("model"))
    DataLoader = tfbp.get_data_loader(config.get("data_loader"))

    train_method = config.get("train_method")
    eval_method, direction = config.get("eval_method")
    objective = kt.Objective(eval_method, direction)

    hparams = kt.HyperParameters()
    all_hp = kt.HyperParameters()
    Model.hparams(all_hp)
    DataLoader.hparams(all_hp)
    for p in all_hp.space:
        if p.name in config.get("excluding"):
            hparams.Fixed(p.name, p.default)

    Model.hparams(hparams)
    DataLoader.hparams(hparams)

    # TODO: add support for hyperband.
    algorithm = config.get("algorithm")
    if algorithm == "bayes":
        max_trials = config.get("max_trials")
        oracle = kt.oracles.BayesianOptimization(
            objective,
            max_trials,
            hyperparameters=hparams,
            num_initial_points=config.get("num_initial_points"),
            alpha=config.get("alpha"),
            beta=config.get("beta"),
        )

    elif algorithm == "random":
        max_trials = config.get("max_trials")
        oracle = kt.oracles.RandomSearch(objective, max_trials, hyperparameters=hparams)

    tuner = tfbp.Tuner(
        oracle,
        Model,
        save_dir=save_dir,
        train_method=train_method,
        eval_method=eval_method,
    )
    tuner.search(DataLoader)

    print("**Best trials**\n\nrank\ttrial_id\tscore")
    for i, t in enumerate(oracle.get_best_trials(num_trials=None), 1):
        print("{}\t{}\t{:.4f}".format(i, t.trial_id, t.score))


if __name__ == "__main__":
    usage_str = (
        "Usage:\n  New run: python tune.py [save_dir] [model] [data_loader] -t "
        "[train_method] -e [eval_method] [min|max] -a [algorithm] [tuning_options...]?\n"
        " Existing run: python tune.py [save_dir]"
    )
    if len(sys.argv) < 2:
        print(usage_str, file=sys.stderr)
        exit(1)

    save_dir = os.path.join("experiments", sys.argv[1])
    # Avoid errors due to a missing directories.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Dynamically parse arguments from the command line depending on whether this is a
    # new or existing run. The `save_dir` arguments is always required.
    parser = ArgumentParser()
    parser.add_argument("save_dir", type=str)

    # If tune_metadata.json exists, all the tuning config can be inferred.
    warn_ignoring = False
    tune_metadata_path = os.path.join(save_dir, "tune_metadata.json")
    tune_metadata = None
    if os.path.isfile(tune_metadata_path):

        with open(tune_metadata_path) as f:
            tune_metadata = json.load(f)

        # The model and data loader shouldn't be provided for an existing run, but for
        # convenience this error is handled for the user.
        if len(sys.argv) > 2 and not sys.argv[2].startswith("-"):
            warn_ignoring = True
            parser.add_argument("model", type=str)
            if len(sys.argv) > 3 and not sys.argv[3].startswith("-"):
                parser.add_argument("data_loader", type=str)

    else:
        parser.add_argument("model", type=str)
        parser.add_argument("data_loader", type=str)

    # Model runnables for tuning.
    parser.add_argument("-t", "--train_method", type=str)
    parser.add_argument("-e", "--eval_method", type=str, nargs="+")

    # Blacklist / whitelist of hyperparameters.
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--include", type=str, nargs="+")
    group.add_argument("--exclude", type=str, nargs="+")

    # Tuning strategy.
    # https://keras-team.github.io/keras-tuner/documentation/oracles/
    parser.add_argument("-a", "--algorithm")

    # TODO: add support for hyperpand.
    m = re.search(r"(?:-a|--algorithm)(?:=| +)([a-z]+)", " ".join(sys.argv))
    if m and m.group(1) == "bayes":
        parser.add_argument("--max_trials", type=int, default=10)
        parser.add_argument("--num_initial_points", type=int, default=3)
        parser.add_argument("--alpha", type=float, default=1e-4)
        parser.add_argument("--beta", type=int, default=2.6)

    elif m and m.group(1) == "random":
        parser.add_argument("--max_trials", type=int, default=10)

    elif m:
        raise ValueError(f"Tuning algorithm not supported: {m.group(1)}")

    FLAGS = parser.parse_args()

    if tune_metadata is None:
        if (
            FLAGS.train_method is None
            or FLAGS.eval_method is None
            or len(FLAGS.eval_method) == 1
        ):
            print(usage_str, file=sys.stderr)
            exit(1)

        # Get the hyperparameters to exclude from the search.
        excluding = []
        khp = kt.HyperParameters()
        Model = tfbp.get_model(FLAGS.model)
        DataLoader = tfbp.get_data_loader(FLAGS.data_loader)
        Model.hparams(khp)
        DataLoader.hparams(khp)
        for p in khp.values.keys():
            if FLAGS.include is not None:
                if p not in FLAGS.include:
                    excluding.append(p)
            elif FLAGS.exclude is not None:
                if p in FLAGS.exclude:
                    excluding.append(p)
            else:
                excluding.append(p)

        tune_metadata = {
            "model": FLAGS.model,
            "data_loader": FLAGS.data_loader,
            "train_method": FLAGS.train_method,
            "eval_method": FLAGS.eval_method,
            "algorithm": FLAGS.algorithm,
            "excluding": excluding,
        }

        # TODO: add support for hyperband.
        if FLAGS.algorithm == "bayes":
            tune_metadata["max_trials"] = FLAGS.max_trials
            tune_metadata["num_initial_points"] = FLAGS.num_initial_points
            tune_metadata["alpha"] = FLAGS.alpha
            tune_metadata["beta"] = FLAGS.beta

        elif FLAGS.algorithm == "random":
            tune_metadata["max_trials"] = FLAGS.max_trials

        else:
            print(usage_str, file=sys.stderr)
            exit(1)

        with open(tune_metadata_path, "w") as f:
            json.dump(
                tune_metadata, f, indent=4, sort_keys=True,
            )

    # TODO: add redundancy checks for specific algorithm parameters.
    elif not (
        FLAGS.train_method is None
        and FLAGS.eval_method is None
        and FLAGS.include is None
        and FLAGS.exclude is None
    ):
        warn_ignoring = True

    if warn_ignoring:
        print(
            f"Warning: tuning configuration already specified at {save_dir},",
            "ignoring some args...",
            file=sys.stderr,
        )

    search_hyperparameters(save_dir, tune_metadata)
