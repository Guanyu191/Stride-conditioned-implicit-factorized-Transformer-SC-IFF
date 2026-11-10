"""Read this file first: Stage 1 -> VALID selection -> TEST -> Stage 2 -> VALID selection -> TEST."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time

# Needed for deterministic CUDA linear algebra; set before PyTorch starts.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from config import settings
from data import open_data, coordinates, fit_statistics, origins, two_step_batch, trajectory_batch
from model import Predictor
from fusion import Fusion
from losses import two_step_loss, trajectory_loss
from rollout import training_rollout
from evaluate import references, validate, short_validation, test
from metrics import choose


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2))


def save_checkpoint(path, model, optimizer, scheduler, step, history):
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict() if scheduler else None, step=step, history=history), path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def restore(path, model, optimizer, scheduler=None):
    """Resume from the last saved checkpoint, if there is one."""
    if not path.exists():
        return 0, []
    checkpoint = load_checkpoint(path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint["step"], checkpoint["history"]


def gradient_step(loss, parameters, optimizer, threshold):
    """Backpropagate, clip by the global norm and update."""
    loss.backward()
    gradients = [p.grad for p in parameters if p.grad is not None]
    # Sum the norm in float64 because long feedback chains can overflow float32.
    norm = torch.stack([g.detach().double().norm() for g in gradients]).norm()
    factor = (threshold / (norm + 1e-6)).clamp(max=1)
    for gradient in gradients:
        gradient.mul_(factor.to(gradient.dtype))
    optimizer.step()
    return float(norm)


def select_and_restore(folder, model, history):
    """Load the candidate chosen on VALID."""
    selected, rule = choose([r for r in history if r["valid"]])
    model.load_state_dict(load_checkpoint(folder / f"candidate_{selected['step']:04d}.pt")["model"])
    write_json(folder / "selection.json", dict(selected=selected, rule=rule))
    return model.eval()


def train_stage1(cfg, data, stats, positions, reference, folder, kind, stride, selected_parents):
    """Train one predictor on two-step rollouts and return its selected epoch.

    IFF and the shared Concat/FiLM predictors use one call. Indep calls this once
    per stride, from coarse to fine, with the predictors already selected.
    """
    seed_everything(cfg.seed)
    model = Predictor(cfg, stats, kind).to(cfg.device)
    if selected_parents:
        # Each finer Indep predictor starts from the selected coarser one.
        model.load_state_dict(selected_parents[min(selected_parents)].state_dict())
    folder.mkdir(parents=True, exist_ok=True)

    scales = (stride,) if kind in ("indep", "iff") else cfg.strides
    table = origins(cfg, scales, cfg.train_ids)
    batch, epochs = cfg.stage1_batch, cfg.epochs(kind, stride)
    steps = len(table) // batch
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.stage1_lr, weight_decay=cfg.stage1_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps * epochs, eta_min=cfg.stage1_final_lr)
    start, history = restore(folder / "last.pt", model, optimizer, scheduler)

    for epoch in range(start, epochs):
        model.train()
        order = np.random.RandomState(cfg.seed + epoch).permutation(len(table))
        tick, losses = time.monotonic(), []
        for step in range(steps):
            rows = table[order[step * batch:(step + 1) * batch]]
            values, strides = two_step_batch(data, rows, cfg.device)
            optimizer.zero_grad(set_to_none=True)
            loss = two_step_loss(model, values, strides, positions)
            gradient_step(loss, list(model.parameters()), optimizer, cfg.stage1_clip)
            scheduler.step()
            losses.append(float(loss.detach()))
            if step % 100 == 0:
                print(json.dumps(dict(stage=1, stride=stride, epoch=epoch + 1, epochs=epochs, step=step + 1, steps=steps, loss=losses[-1])), flush=True)

        # Score this epoch with free-running prediction on VALID.
        model.eval()
        parents = dict(selected_parents)
        parents.update({s: model for s in scales})
        row = dict(step=epoch + 1, train_loss=float(np.mean(losses)), train_seconds=time.monotonic() - tick)
        try:
            row.update(validate(parents, data, cfg, positions, reference, stride if kind == "indep" else 1))
            row.update(short_loss=short_validation(model, data, cfg, positions, scales), valid=True)
        except FloatingPointError as error:
            row.update(valid=False, error=str(error))  # A diverged epoch cannot be selected.
        history.append(row)
        save_checkpoint(folder / f"candidate_{epoch + 1:04d}.pt", model, optimizer, scheduler, epoch + 1, history)
        save_checkpoint(folder / "last.pt", model, optimizer, scheduler, epoch + 1, history)
        write_json(folder / "history.json", history)
        print(json.dumps(row), flush=True)
    return select_and_restore(folder, model, history)


def train_stage2(cfg, data, stats, positions, reference, folder, parents):
    """Freeze the predictors, train only FA-CSF on long rollouts and return the selected update."""
    seed_everything(cfg.fusion_seed)
    model = Fusion(cfg, stats, parents).to(cfg.device).train()
    folder.mkdir(parents=True, exist_ok=True)
    parameters = model.parameters_to_train()
    optimizer = torch.optim.AdamW(parameters, lr=cfg.stage2_lr, weight_decay=cfg.stage2_decay)
    start, history = restore(folder / "last.pt", model, optimizer)
    scale = torch.tensor(stats["loss_scale"], device=cfg.device)

    for step in range(start, cfg.stage2_updates):
        rng = np.random.RandomState(cfg.fusion_seed + step)  # The samples do not depend on resuming.
        initial, truth, indices = trajectory_batch(data, cfg, rng)
        optimizer.zero_grad(set_to_none=True)
        tick = time.monotonic()
        prediction = training_rollout(model, initial, positions, cfg.window)
        loss, terms = trajectory_loss(prediction, truth, scale, cfg)
        grad = gradient_step(loss, parameters, optimizer, cfg.stage2_clip)
        row = dict(step=step + 1, loss=float(loss.detach()), terms={k: float(v.detach()) for k, v in terms.items()},
                   gradient=grad, seconds=time.monotonic() - tick, origins=indices, valid=False)
        del prediction, initial, truth, loss, terms  # Free the GPU memory before VALID.

        evaluate_now = (step + 1) % cfg.stage2_every == 0
        if evaluate_now:
            model.eval()
            try:
                row.update(validate({}, data, cfg, positions, reference, fusion=model))
                row.update(short_loss=short_validation(None, data, cfg, positions, (1,), fusion=model), valid=True)
            except FloatingPointError as error:
                row.update(error=str(error))  # A diverged update cannot be selected.
            model.train()
        history.append(row)
        if evaluate_now:
            save_checkpoint(folder / f"candidate_{step + 1:04d}.pt", model, optimizer, None, step + 1, history)
        save_checkpoint(folder / "last.pt", model, optimizer, None, step + 1, history)
        write_json(folder / "history.json", history)
        print(json.dumps(row), flush=True)
    return select_and_restore(folder, model, history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("iff", "concat", "indep", "film"))
    args = parser.parse_args()
    cfg = settings
    seed_everything(cfg.seed)

    # Data, TRAIN statistics and VALID references, shared by all models.
    data = open_data(cfg)
    root = Path(cfg.output_root)
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "statistics.json").exists():
        write_json(root / "statistics.json", fit_statistics(data, cfg))
    stats = json.loads((root / "statistics.json").read_text())
    positions = coordinates(cfg)
    reference = references(data, cfg, root / "validation_references")
    folder = root / args.model
    folder.mkdir(exist_ok=True)
    write_json(folder / "settings.json", asdict(cfg))

    # Stage 1: `parents` maps each stride to its predictor for hierarchical prediction.
    if args.model == "indep":
        parents = {}
        for stride in reversed(cfg.strides):
            parents[stride] = train_stage1(cfg, data, stats, positions, reference, folder / f"stage1_s{stride}", args.model, stride, parents)
        registered = {str(s): m for s, m in parents.items()}
    else:
        model = train_stage1(cfg, data, stats, positions, reference, folder / "stage1", args.model, 1, {})
        parents = {s: model for s in ((1,) if args.model == "iff" else cfg.strides)}
        registered = {"shared": model}
    if not (folder / "test_native/summary.json").exists():
        test(parents, data, cfg, positions, reference, folder / "test_native")

    # Stage 2: FA-CSF on the frozen SC-IFF predictors. IFF has no Stage 2.
    if args.model != "iff":
        fusion = train_stage2(cfg, data, stats, positions, reference, folder / "stage2", registered)
        test({}, data, cfg, positions, reference, folder / "test_fa_csf", fusion=fusion)


if __name__ == "__main__":
    main()
