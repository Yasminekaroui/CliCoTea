import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from lavis.models import load_model_and_preprocess
from lavis.tasks import RetrievalTask
from clicotea.options import evaluate_options
from clicotea.data import (
    VEDataset,
    VRDataset,
    RetrievalDataset,
)
from clicotea.model import TokenAlignmentModel
from clicotea.utils import setup_logger


def compute_accuracy(loader, model):
    # start evaluation
    acc = 0
    n_samples = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = {
                k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            output = model(batch, is_train=False)
            # compute accuracy
            acc += (
                torch.argmax(output["predictions"], dim=1)
                .eq(output["targets"])
                .sum()
                .item()
            )
            n_samples += len(output["targets"])
    acc /= n_samples
    return acc


def evaluate(args):
    # setup logger
    output_dir = os.path.dirname(args.text_encoder_ckpt)
    logger = setup_logger(args.task, args.log, output_dir, "test.log")

    logger.info(f"Loading model from {args.model_name}")
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name=args.model_name,
        model_type=args.model_type,
        is_eval=True,
        device=args.device,
    )
    # load checkpoint from multilingual text encoder
    logger.info(f"Loading checkpoint from {args.text_encoder_ckpt}")
    # initialize the VL text encoder model with the multilingual text encoder
    TokenAlignmentModel.init_from_student_model(
        model, torch.load(args.text_encoder_ckpt)["model"]
    )

    # define datasets
    if args.task == "vr":
        dset = VRDataset(
            ann_file=args.test_file,
            image_root=args.marvl_image_root,
            vis_processor=vis_processors["eval"],
            txt_processor=txt_processors["eval"],
        )
    elif args.task == "ve":
        dset = VEDataset(
            ann_file=args.test_file,
            image_root=args.flickr30k_image_root,
            vis_processor=vis_processors["eval"],
            txt_processor=txt_processors["eval"],
        )
    elif args.task == "retrieval":
        dset = RetrievalDataset(
            ann_file=args.test_file,
            flickr30k_image_root=args.flickr30k_image_root,
            coco_image_root=args.coco_image_root,
            vis_processor=vis_processors["eval"],
            txt_processor=txt_processors["eval"],
        )

    # define dataloader
    loader = DataLoader(
        dset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )

    if args.task == "vr" or args.task == "ve":
        acc = compute_accuracy(loader, model)
        logger.info(f"Accuracy: {acc*100:.2f}%")
    elif args.task == "retrieval":
        task = RetrievalTask(args)
        results = task.evaluation(model, loader)
        logger.info(results)


if __name__ == "__main__":
    args = evaluate_options()
    evaluate(args)
