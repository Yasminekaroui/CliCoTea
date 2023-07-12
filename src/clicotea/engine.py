# encoding: utf-8
"""
@author:  Remi Lebret
@contact: remi@lebret.ch
"""
import math
import torch
from torch.nn.utils import clip_grad_norm_

from ignite.engine import Engine


def create_teacher_student_trainer(
    model,
    optimizer,
    loss_fn,
    accumulation_steps: int = 1,
    gradient_clip=math.inf,
    prepare_batch=lambda x: x,
    device=None,
    output_transform=lambda loss: loss.item(),
):
    """
    Factory function for creating a trainer for supervised models.
    Args:
        model (`torch.nn.Module`): the model to train.
        optimizer (`torch.optim.Optimizer`): the optimizer to use.
        loss_fn (torch.nn loss function): the loss function to use.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU,
            the copy may occur asynchronously with respect to the host. For other cases,
            this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`,
            `non_blocking` and outputs tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y',
            'y_pred','loss' and returns value to be assigned to engine's state.output
            after each iteration. Default is returning `loss.item()`.
    Note: `engine.state.output` for this engine is defind by `output_transform`
        parameter and is the loss of the processed batch by default.
    Returns:
        Engine: a trainer engine with supervised update function.
    """
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        src_outputs, tgt_outputs = model(prepare_batch(batch, device=device))
        loss = loss_fn(src_outputs, tgt_outputs) / accumulation_steps
        loss.backward()

        if engine.state.iteration % accumulation_steps == 0:
            clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            optimizer.zero_grad()

        return output_transform(loss)

    return Engine(_update)


def create_teacher_student_evaluator(
    model,
    metrics=None,
    device=None,
    non_blocking=False,
    prepare_batch=None,
    output_transform=lambda x, y: (x, y),
):
    """
    Factory function for creating an evaluator for supervised models.

    Args:
        model (`torch.nn.Module`): the model to train.
        metrics (dict of str - :class:`~ignite.metrics.Metric`): a map of metric names
            to Metrics.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU,
            the copy may occur asynchronously with respect to the host. For other cases,
            this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`,
            `non_blocking` and outputs tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred'
            and returns value to be assigned to engine's state.output after each
            iteration. Default is returning `(y_pred, y,)` which fits output expected
            by metrics. If you change it you should use `output_transform` in metrics.

    Note: `engine.state.output` for this engine is defind by `output_transform`
        parameter and is a tuple of `(batch_pred, batch_y)` by default.

    Returns:
        Engine: an evaluator engine with supervised inference function.
    """
    metrics = metrics or {}

    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = model(prepare_batch(batch, device=device))
            return output_transform(x, y)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine
