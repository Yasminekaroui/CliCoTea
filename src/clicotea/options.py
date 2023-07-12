# encoding: utf-8
"""
@author:  Remi Lebret
@contact: remi@lebret.ch
"""
import argparse

from clicotea.solver import OPTIMIZER


def _io_arguments(parser):
    group = parser.add_argument_group("I/O options")
    group.add_argument(
        "--output-dir",
        default="exp",
        help="directory where to save stuff",
    )
    group.add_argument("--batch-size", help="mini-batch size", default=32, type=int)
    group.add_argument(
        "--num-workers", help="number of workers for data loading", default=8, type=int
    )
    group.add_argument(
        "--log",
        help="logging level",
        default=20,
        type=int,
        choices=[0, 10, 20, 30, 40, 50],
    )
    group.add_argument("--device", default="cuda", help="name of the device", type=str)
    return group


def _training_option_arguments(parser):
    group = parser.add_argument_group("Training options")
    group.add_argument("--epoch", help="number of epoch", default=25, type=int)
    group.add_argument("--epoch-size", help="epoch size", default=None, type=int)
    group.add_argument("--seed", help="random seed (default: 1)", default=1, type=int)
    group.add_argument(
        "--early-stopping",
        help="enables early stopping",
        action="store_true",
        default=False,
    )
    group.add_argument(
        "--stop-after",
        help="when do stop training when no improvement",
        default=4,
        type=int,
    )
    group.add_argument("--n-saved", help="number of model to save", default=2, type=int)
    group.add_argument("--log-every-iters", help="when to log", default=100, type=int)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a "
        "backward/update pass.",
    )
    return group


def _solver_arguments(parser):
    group = parser.add_argument_group("Optimization options")
    group.add_argument(
        "--optim", help="optimizer algorithm", default="adam", choices=OPTIMIZER.keys()
    )
    group.add_argument(
        "--lr-scheduler",
        help="method to adjust the learning rate based on the number of epochs. "
        "(default: none)",
        default="none",
        choices=["fixed", "linear", "none"],
    )
    group.add_argument("--lr", default=5e-5, type=float, help="learning rate")
    group.add_argument(
        "--warmup-updates",
        default=0,
        type=int,
        metavar="N",
        help="number of steps for warmup",
    )
    group.add_argument(
        "--total-num-updates",
        default=0,
        type=int,
        metavar="N",
        help="total number of steps",
    )
    group.add_argument(
        "--max-grad-norm", default=1.2, type=float, help="Max gradient norm."
    )
    return group


def _wandb_arguments(parser):
    group = parser.add_argument_group("WandB tracking options")
    group.add_argument(
        "--exp-name",
        default="token-align",
        type=str,
        help="WandB tracking project name",
    )
    group.add_argument(
        "--run-name", help="WandB tracking run name", default="", type=str
    )
    return group


def train_token_alignment_options():
    parser = argparse.ArgumentParser(
        description="PyTorch Teacher Learning for Contextualised Token Alignment"
    )
    parser.add_argument(
        "--train-file",
        default="",
        help="path to the training file with parallel corpus for alignment",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dev-file",
        default="",
        help="path to the dev file with parallel corpus for alignment",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--teacher-model-name",
        default="albef_pretrain",
        help="name of Vision-Language teacher model",
        type=str,
        choices=["albef_retrieval", "albef_nlvr", "albef_classification"],
        required=True,
    )
    parser.add_argument(
        "--teacher-model-type",
        default="flickr",
        help="type of Vision-Language teacher model",
        type=str,
        choices=["flickr", "coco", "ve", "nlvr"],
        required=True,
    )
    parser.add_argument(
        "--student-model-name",
        default="bert-base-multilingual-cased",
        help="name of student model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num-layers", default=1, help="align the last n layers", type=int
    )
    parser.add_argument(
        "--add-en-sent",
        help="learning with English sentences",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--align-with-cls", help="align CLS token", action="store_true", default=False
    )
    _io_arguments(parser)
    _training_option_arguments(parser)
    _wandb_arguments(parser)
    _solver_arguments(parser)
    args = parser.parse_args()
    print(args)
    return args


def evaluate_options():
    parser = argparse.ArgumentParser(
        description="PyTorch Zero-Shot Evalution of CLiCoTEA"
    )
    parser.add_argument(
        "--test-file",
        default="",
        help="path to the test file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--task",
        default="vr",
        help="name of the task to evaluate",
        type=str,
        choices=["vr", "ve", "retrieval"],
        required=True,
    )
    parser.add_argument(
        "--flickr30k-image-root",
        default=None,
        help="path the Flickr30k images",
        type=str,
    )
    parser.add_argument(
        "--coco-image-root",
        default=None,
        help="path the COCO images",
        type=str,
    )
    parser.add_argument(
        "--marvl-image-root",
        default=None,
        help="path the MaRVL images",
        type=str,
    )
    parser.add_argument(
        "--model-name",
        default="albef_pretrain",
        help="name of Vision-Language teacher model",
        type=str,
        choices=["albef_retrieval", "albef_nlvr", "albef_classification"],
        required=True,
    )
    parser.add_argument(
        "--model-type",
        default="flickr",
        help="type of Vision-Language teacher model",
        type=str,
        choices=["flickr", "coco", "ve", "nlvr"],
        required=True,
    )
    parser.add_argument(
        "--text-encoder-name",
        help="name of the multilingual text encoder",
        default="bert-base-multilingual-cased",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--text-encoder-ckpt",
        help="path to the multilingual text encoder checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--k-test",
        help="number of retrieval candidates ranked from contrastive similarity",
        type=int,
        default=256,
    )
    _io_arguments(parser)
    args = parser.parse_args()
    print(args)
    return args
