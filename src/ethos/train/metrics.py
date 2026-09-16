import math
import os
from xml.parsers.expat import model

import torch as th
from torch.utils.data import DataLoader
from transformers import PreTrainedModel


@th.inference_mode()
def estimate_loss(
    model: PreTrainedModel, ctx, loaders: list[tuple[str, DataLoader]], eval_iters: int
) -> dict:
    rank = int(os.environ.get("RANK", -1))
    is_distributed = rank != -1

    if is_distributed:
        eval_iters = math.ceil(eval_iters / int(os.environ["WORLD_SIZE"]))
        
    device = next(model.parameters()).device

    out = {}
    for split, dataloader in loaders:
        losses = th.empty(eval_iters, device=next(model.parameters()).device)
        for i, (X, Y) in zip(range(eval_iters), dataloader):
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            with ctx:
                
                if isinstance(X, tuple):
                    output = model(input_ids=X[0], decoder_input_ids=X[1], labels=Y)
                else:
                    output = model(input_ids=X, labels=Y)
                
                loss = output.loss
            losses[i] = loss.item()

        out[f"loss/{split}"] = losses.mean().item()
    return out
