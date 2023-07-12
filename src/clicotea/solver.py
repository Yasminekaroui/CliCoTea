# encoding: utf-8
"""
@author:  Remi Lebret
@contact: remi@lebret.ch
"""

from torch.optim import SGD, Adagrad, Adadelta, AdamW, Adam

from transformers.optimization import get_constant_schedule
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.optimization import get_constant_schedule_with_warmup


OPTIMIZER = {
    "adam": Adam,
    "adamW": AdamW,
    "sgd": SGD,
    "adadelta": Adadelta,
    "adagrad": Adagrad,
}


def set_optimizer(args, model):
    return OPTIMIZER[args.optim](
        params=model.parameters(),
        lr=args.lr,
    )


def set_scheduler(args, optimizer, logger):
    if args.lr_scheduler == "fixed":
        return get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_updates
        )
    elif args.lr_scheduler == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_updates,
            num_training_steps=args.total_num_updates,
        )
    else:
        logger.info(
            f"{args.lr_scheduler} is not available in Transformers, "
            "using constant scheduler instead"
        )
        return get_constant_schedule(optimizer)
