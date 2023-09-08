import argparse

from .baselines import Baseline


def get_args():
    parser = argparse.ArgumentParser(description="Bongard-Context")
    parser.add_argument("data", default="", help="path to dataset root")
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--runs_dir", type=str, default="./runs/", help="path to Tensorboard runs"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./checkpoints/", help="path to checkpoints"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results/",
        help="path to evaluation results, used only at test time",
    )
    parser.add_argument(
        "--id",
        type=str,
        default="",
        help="to help disambiguate training runs, can specify an auxiliary id to append to the name",
    )
    parser.add_argument(
        "--load_path",
        default="",
        type=str,
        help="load from a checkpoint. At train time, loads only encoder. At test time, loads whole model.",
    )
    parser.add_argument(
        "--load_encoder",
        default="",
        type=str,
        help="path to encoder to load. Used only at test time.",
    )
    parser.add_argument(
        "--train_steps", default=30000, type=int, help="number of train iterations"
    )
    parser.add_argument(
        "--eval_every",
        default=1000,
        type=int,
        help="how often to evaluate on the validation set",
    )
    parser.add_argument(
        "--save_every", default=10000, type=int, help="how often to save checkpoints"
    )
    parser.add_argument(
        "--train_subset",
        default=1.0,
        type=float,
        help="for debugging purposes, can specify a percent subset of the training set to use",
    )
    parser.add_argument(
        "--val_subset",
        default=1.0,
        type=float,
        help="if the validation set is large, can specify a percent subset to use",
    )

    # Hyperparameters
    parser.add_argument("--resolution", default=224, type=int, help="image resolution")
    parser.add_argument("-b", "--batch_size", default=1, type=int, help="batch size")
    parser.add_argument(
        "--lr",
        "--learning_rate",
        default=5e-6,
        type=float,
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay")
    parser.add_argument(
        "--use_scheduler",
        action="store_true",
        default=False,
        help="whether to use a OneCycle LR scheduler",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="for the contrastive loss"
    )
    parser.add_argument("--kernel", type=str, default="linear", help="SVM kernel kind")
    parser.add_argument("--k", type=int, default=3, help="k for KNN baseline")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="sets the random seed ONLY for test-time robustness evaluation in paper plots",
    )

    # Data work
    parser.add_argument(
        "--dataset", type=str, default="hoi", help="which dataset to use"
    )
    parser.add_argument(
        "--use_augs",
        action="store_true",
        default=False,
        help="whether or not to use data augmentation",
    )
    parser.add_argument(
        "--dropout_support_prob",
        default=0.0,
        type=float,
        help="the percentage of train batches to which to apply support dropout, used only at train time",
    )
    parser.add_argument(
        "--label_noise_support_prob",
        default=0.0,
        type=float,
        help="the percentage of train batches to which to apply label noise, used only at train time",
    )
    parser.add_argument(
        "--balance_dataset",
        action="store_true",
        default=False,
        help="whether or not to train with our cleaned HOI datasets (see paper), can only be specified at train time",
    )

    # Additional model configuration
    parser.add_argument("-a", "--arch", default="clip", help="encoder architecture")
    parser.add_argument(
        "--train_encoder",
        action="store_true",
        default=False,
        help="whether to train/fine-tune the encoder",
    )
    parser.add_argument(
        "--use_pmf",
        action="store_true",
        default=False,
        help="whether to use PMF instead of support-set Transformers",
    )
    parser.add_argument(
        "--mimic_kind", type=str, default="SVM", help="the kind of 'rule' to mimic"
    )

    # For evaluation time only
    parser.add_argument(
        "--baseline",
        type=str,
        default="NONE",
        help="the baseline method to use for evaluation",
    )
    parser.add_argument(
        "--eval_dropout_support",
        type=int,
        default=-1,
        help="the exact number of supports to retain for each Bongard problem at test time. If -1, all are retained.",
    )
    parser.add_argument(
        "--eval_label_noise",
        type=int,
        default=-1,
        help="the exact number of supports to noise for each Bongard problem at test time. If -1, none is noised.",
    )
    parser.add_argument(
        "--eval_on_val",
        default=False,
        action="store_true",
        help="for debugging and cross-validation, can specify whether to evaluate on validation set rather than test set",
    )
    parser.add_argument(
        "--eval_standardize",
        action="store_true",
        default=False,
        help="whether to apply support-set standardization, as described in the paper",
    )
    parser.add_argument(
        "--eval_standardize_train",
        default="",
        type=str,
        help="path to whole-training-set statistics to perform train-set standardization, as described in the paper. If no path is specified, train-set standardization is not used.",
    )
    parser.add_argument(
        "--eval_normalize_l2",
        action="store_true",
        default=False,
        help="whether to apply l2-normalization, as described in the paper",
    )

    args = parser.parse_args()
    args.baseline = getattr(Baseline, args.baseline)

    return args
