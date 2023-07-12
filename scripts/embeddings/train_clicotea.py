# encoding: utf-8
"""
@author:  Remi Lebret
@contact: remi@lebret.ch
"""
from pathlib import Path

from clicotea.trainer import train_token_alignment
from clicotea.options import train_token_alignment_options
from clicotea.utils import setup_logger
from clicotea.data import TokenAlignmentDataset
from clicotea.model import TokenAlignmentModel


def train(args):
    # create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # setup logger
    logger = setup_logger(args.exp_name, args.log, args.output_dir, "train.log")

    # define model
    model = TokenAlignmentModel(
        teacher_model_name=args.teacher_model_name,
        teacher_model_type=args.teacher_model_type,
        student_model_name=args.student_model_name,
        num_layers=args.num_layers,
        device=args.device,
    )
    # define datasets
    train_dset = TokenAlignmentDataset(
        token_alignment_file=args.train_file,
        src_tokenizer_name=model.teacher_model.tokenizer.name_or_path,
        tgt_tokenizer_name=args.student_model_name,
        add_en_sent=args.add_en_sent,
        align_with_cls=args.align_with_cls,
    )
    dev_dset = TokenAlignmentDataset(
        token_alignment_file=args.dev_file,
        src_tokenizer_name=model.teacher_model.tokenizer.name_or_path,
        tgt_tokenizer_name=args.student_model_name,
        add_en_sent=args.add_en_sent,
        align_with_cls=args.align_with_cls,
    )
    # define dataloaders
    train_loader = train_dset.get_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=True,
    )
    dev_loader = dev_dset.get_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=False,
    )
    # start training
    train_token_alignment(args, model, train_loader, dev_loader, logger)


if __name__ == "__main__":
    args = train_token_alignment_options()
    train(args)
