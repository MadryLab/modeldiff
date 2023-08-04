import numpy as np
import torch
import torch.nn.functional as F
from . import train_utils
from . import common_utils
from torch.cuda.amp import autocast
from contextlib import nullcontext

@common_utils.check_not_multihead(0)
def get_accuracy_and_loss(model, loader, device, loss_fn=F.cross_entropy,
                          enable_amp=True, lock=None, use_eval=True, lr_tta=False):
    # loss_fn: any function that takes in (logits, targets) and outputs a scalar
    assert next(model.parameters()).device == device

    in_tr_mode = model.training
    if use_eval: model = model.eval()
    lock = lock if lock is not None else nullcontext()

    acc_meter = train_utils.AverageMeter()
    loss_meter = train_utils.AverageMeter()

    with torch.no_grad():
        for xb, yb, *_ in loader:
            bs = len(xb)
            xb, yb = xb.to(device), yb.to(device)

            with autocast(enabled=enable_amp):
                with lock:
                    out = model(xb)

                    if lr_tta:
                        out += model(torch.fliplr(xb))
                        out /= 2

            preds = out.argmax(-1)

            b_acc = (preds==yb).float().mean().item()
            b_loss = loss_fn(out, yb).item()

            acc_meter.update(b_acc, bs)
            loss_meter.update(b_loss, bs)

            xb, yb = xb.cpu(), yb.cpu()

    if in_tr_mode:
        model.train()

    return {
        'acc': acc_meter.mean(),
        'loss': loss_meter.mean()
    }

@common_utils.check_not_multihead(0)
def get_predictions(model, loader, device, enable_amp=True, lock=None, use_eval=True, lr_tta=False):
    assert next(model.parameters()).device == device
    in_tr_mode = model.training
    if use_eval: model = model.eval()
    lock = lock if lock is not None else nullcontext()
    preds = []
    labels = []

    with torch.no_grad():
        for xb, yb, *_ in loader:
            xb = xb.to(device)

            with autocast(enabled=enable_amp):
                with lock:
                    out = model(xb)
                    if lr_tta:
                        out += model(torch.fliplr(xb))
                        out /= 2

            yh = out.argmax(-1).cpu().numpy()
            preds.append(yh)

            yb = yb.clone().cpu().numpy()
            labels.append(yb)

            xb = xb.cpu()

    if in_tr_mode:
        model.train()

    preds, labels = map(np.concatenate, [preds, labels])
    return preds, labels

@common_utils.check_not_multihead(0)
def get_margins(model, loader, device, enable_amp=True, lock=None, use_eval=True, lr_tta=False):
    assert next(model.parameters()).device == device
    in_tr_mode = model.training
    if use_eval: model = model.eval()
    all_margins = []
    lock = lock if lock is not None else nullcontext()

    with torch.no_grad():
        for xb, yb, *_ in loader:
            xb = xb.to(device, non_blocking=True)
            rng = torch.arange(len(xb))

            with autocast(enabled=enable_amp):
                with lock:
                    out = model(xb)
                    if lr_tta:
                        out += model(torch.fliplr(xb))
                        out /= 2

            class_logits = out[rng, yb].clone()
            out[rng, yb] = -np.inf
            max_wo_class = out[rng, out.argmax(1)]
            class_logits = (class_logits - max_wo_class).cpu()
            all_margins.append(class_logits)

    if in_tr_mode:
        model = model.train()

    all_margins = torch.cat(all_margins).numpy()
    return all_margins

@common_utils.check_not_multihead(0)
def get_logits(model, loader, device, enable_amp=True, lock=None, apply_fn=None, use_eval=True, lr_tta=False, with_labels=False):
    assert next(model.parameters()).device == device
    in_tr_mode = model.training
    if use_eval: model = model.eval()
    lock = lock if lock is not None else nullcontext()
    all_logits = []
    labels = []

    with torch.no_grad():
        for xb, yb, *_ in loader:
            if apply_fn: xb = apply_fn(xb)
            xb = xb.to(device, non_blocking=True)

            with autocast(enabled=enable_amp):
                with lock:
                    logits = model(xb)
                    if lr_tta:
                        logits += model(torch.fliplr(xb))
                        logits /= 2
                    logits = logits.clone().cpu()
            all_logits.append(logits)

            yb = yb.clone().cpu().numpy()
            labels.append(yb)

    if in_tr_mode:
        model = model.train()

    all_logits = torch.cat(all_logits).numpy()
    labels = np.concatenate(labels)

    if with_labels: return all_logits, labels
    return all_logits