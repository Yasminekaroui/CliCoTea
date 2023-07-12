# encoding: utf-8
"""
@author:  Remi Lebret
@contact: remi@lebret.ch
"""

import torch
import torch.nn.functional as F

from ignite.engine import Events
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
    global_step_from_engine,
    Timer,
    TerminateOnNan,
)
from ignite.metrics import RunningAverage, MeanAbsoluteError, MeanSquaredError
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.engines import common

from clicotea.solver import set_optimizer, set_scheduler
from clicotea.engine import (
    create_teacher_student_trainer,
    create_teacher_student_evaluator,
)
from clicotea.data.token_alignment_dataset import prepare_token_alignment_batch


def train_token_alignment(
    args,
    model,
    train_loader,
    dev_loader,
    logger,
):
    loss_fn = F.mse_loss
    n_saved = args.n_saved
    device = args.device
    epochs = args.epoch
    output_dir = args.output_dir
    max_grad_norm = args.max_grad_norm
    gradient_accumulation_steps = args.gradient_accumulation_steps
    exp_name = args.exp_name if args.exp_name else None
    run_name = args.run_name if args.run_name else None
    log_every_iters = args.log_every_iters

    logger.info("Start training")

    # set an optimizer algorithm
    optimizer = set_optimizer(args, model)
    # with a scheduler
    lr_scheduler = set_scheduler(args, optimizer, logger)

    # create trainer
    trainer = create_teacher_student_trainer(
        model,
        optimizer,
        loss_fn,
        accumulation_steps=gradient_accumulation_steps,
        gradient_clip=max_grad_norm,
        prepare_batch=prepare_token_alignment_batch,
        device=device,
    )
    # computer average of the loss and attach it to trainer
    RunningAverage(output_transform=lambda x: x).attach(trainer, "avg_loss")

    trainer._logger = logger

    # add checkpoint event
    checkpointer = Checkpoint(
        to_save={
            "model": model,
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "trainer": trainer,
        },
        save_handler=DiskSaver(output_dir, create_dir=True),
        n_saved=n_saved,
        global_step_transform=global_step_from_engine(trainer),
    )

    # create evaluator
    evaluator = create_teacher_student_evaluator(
        model,
        metrics={
            "mse": MeanSquaredError(
                output_transform=lambda x: (x[0].squeeze(0), x[1].squeeze(0))
            ),
            "mae": MeanAbsoluteError(
                output_transform=lambda x: (x[0].squeeze(0), x[1].squeeze(0))
            ),
        },
        prepare_batch=prepare_token_alignment_batch,
        device=device,
    )
    # attach checkpoint to evaluator
    evaluator.add_event_handler(Events.COMPLETED, checkpointer)
    evaluator._logger = logger

    # attach progress bar
    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=["avg_loss"])
    ProgressBar(persist=False, desc="Validation").attach(evaluator)

    # attach timer to trainer
    timer = Timer(average=True)
    timer.attach(
        trainer,
        start=Events.EPOCH_STARTED,
        resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED,
        step=Events.ITERATION_COMPLETED,
    )

    def lr_scheduling_transformers_iteration(engine):
        lr_scheduler.step()

    # add events to trainer

    # stop the training if the process_function’s output contains a NaN
    # or infinite number
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=gradient_accumulation_steps),
        lr_scheduling_transformers_iteration,
    )

    # Setup MLFlow and TensorBoard logging on evaluators. Logged values are:
    #  - Training metrics, e.g. running average loss values
    #  - Learning rate
    #  - Evaluation test metrics
    # setup tensorboard tracking
    tb_logger = common.setup_tb_logging(
        output_dir, trainer, optimizer, log_every_iters=log_every_iters
    )
    # setup wandb tracking
    wandb_logger = common.setup_wandb_logging(
        trainer,
        optimizer,
        evaluator,
        log_every_iters=log_every_iters,
        project=exp_name,
        name=run_name,
        config={
            "pytorch version": torch.__version__,
            "cuda version": torch.version.cuda,
            "device name": torch.cuda.get_device_name(device),
            "output path": output_dir,
            **vars(args),
        },
    )

    # add events to trainer

    @trainer.on(Events.STARTED)
    def on_training_started(engine):
        pbar.log_message("[start training]")
        pbar.log_message(
            f"\nLaunch TensorBoard with: \ntensorboard --logdir={output_dir}\n\n"
        )

    # attach scheduler to trainer

    @trainer.on(Events.EPOCH_STARTED)
    def on_training_epoch_started(engine):
        pbar.log_message("[training epoch]")

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info(
            "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                engine.state.epoch,
                timer.value() * timer.step_count,
                engine.state.dataloader.batch_size / timer.value(),
            )
        )
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):
        logger.info("[validation epoch]")
        evaluator.run(dev_loader)
        logger.info(
            "Validation Results - Epoch: {}  MSE: {:.2f} MAE: {:.2f}".format(
                engine.state.epoch,
                evaluator.state.metrics["mse"],
                evaluator.state.metrics["mae"],
            )
        )

    @trainer.on(Events.COMPLETED)
    def on_training_completed(engine):
        # close tensorboard summary writer
        tb_logger.close()
        # close wandb logger
        wandb_logger.close()

    # training
    trainer.run(train_loader, max_epochs=epochs)
