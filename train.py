"""OCALD two-stage and weighted joint pipelines with compatible checkpoints."""

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from data import FlightDataset, collate_fn, load_config, load_processed
from losses import head_l2, order_loss, reconstruction_loss, representation_loss, training_loss
from model import OCALD, time_positions
from torch.nn import functional as F


REPRESENTATION_MODULES = ("sensor_encoder", "conditional_prior", "decoder")


def validate_representation_config(config):
    expected = {"optimizer": "Adam", "learning_rate": 0.001, "weight_decay": 0.0,
                "batch_size": 16, "epochs": 100, "sigma_x_squared": 1.0,
                "kl_i_weight": 1.0, "kl_c_weight": 1.0, "gradient_clip_norm": 5.0}
    if config["training"]["representation"] != expected:
        raise ValueError("Stage 1 settings differ from approved v2 values")
    if (config["training"]["seed"] != 42 or config["training"]["device"] != "auto"
            or config["training"]["checkpoint_selection"] != "last_epoch"
            or config["smoke"]["representation_epochs"] != 3
            or config["smoke"]["representation_max_flights"] != 32):
        raise ValueError("Unapproved seed/device/checkpoint/smoke settings")


def representation_state(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            if name.split(".")[0] in REPRESENTATION_MODULES}


def load_representation(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["stage"] != "representation":
        raise ValueError("Expected a representation checkpoint")
    config = checkpoint["config"]
    validate_representation_config(config)
    model = OCALD(config)
    expected = representation_state(model)
    if set(checkpoint["state_dict"]) != set(expected):
        raise ValueError("Missing or unexpected representation weights")
    for name in REPRESENTATION_MODULES:
        prefix = name + "."
        state = {key[len(prefix):]: value for key, value in checkpoint["state_dict"].items() if key.startswith(prefix)}
        getattr(model, name).load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, checkpoint


def train_representation(config, smoke):
    validate_representation_config(config)
    if config['model'].get('score_mode', 'hi_head') != 'hi_head':
        raise ValueError('Scalar I supports joint training only')
    seed = config["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed = load_processed(config)
    dataset = FlightDataset(processed, "dev")
    if len(dataset) == 0:
        raise ValueError("No training flights")
    model = OCALD(config).to(device)
    initial = representation_state(model)
    initial_head = {name: p.detach().clone() for name, p in model.hi_head.named_parameters()}
    for parameter in model.hi_head.parameters():
        parameter.requires_grad_(False)
    indices = list(range(len(dataset)))
    if smoke:
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:min(32, len(dataset))].tolist()
    settings = config["training"]["representation"]
    epochs = config["smoke"]["representation_epochs"] if smoke else settings["epochs"]
    loader = DataLoader(Subset(dataset, indices), batch_size=settings["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed), collate_fn=collate_fn)
    parameters = [p for name in REPRESENTATION_MODULES for p in getattr(model, name).parameters()]
    optimizer = torch.optim.Adam(parameters, lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    mode = "smoke" if smoke else "formal"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(config["output"]["root"]) / f"representation_{mode}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    print(f"Stage=representation mode={mode} device={device} epochs={epochs} flights={len(indices)}/{len(dataset)} batch_size={settings['batch_size']}", flush=True)
    print(f"CPU threads={torch.get_num_threads()}; run_dir={run_dir}", flush=True)
    with (run_dir / "losses.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["stage", "mode", "epoch", "flights", "loss", "reconstruction", "kl_i", "kl_c", "order", "l2"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            model.train()
            totals = {key: 0.0 for key in ("loss", "reconstruction", "kl_i", "kl_c", "order", "l2")}
            count = 0
            for batch in loader:
                x, w, mask = (batch[key].to(device) for key in ("x", "w", "mask"))
                optimizer.zero_grad(set_to_none=True)
                output = model(x, w, mask)
                pair = torch.nonzero((batch['unit'][:, None] == batch['unit'][None, :])
                                     & (batch['cycle'][None, :] - batch['cycle'][:, None] >= 5)).to(device)
                scores = model.hi_head(output['mu_i'])
                losses = training_loss(model.hi_head, representation_weights(config), x=x, mask=mask,
                                       output=output, r_early=scores[pair[:, 0]], r_late=scores[pair[:, 1]],
                                       sigma_x_squared=settings['sigma_x_squared'])
                if not all(torch.isfinite(value).item() for value in losses.values()):
                    raise RuntimeError(f"Nonfinite Stage 1 loss at epoch {epoch}; no configuration changed")
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_clip_norm"], error_if_nonfinite=True)
                optimizer.step()
                size = x.shape[0]
                count += size
                for key in totals:
                    totals[key] += losses[key].detach().item() * size
            row = {"stage": "representation", "mode": mode, "epoch": epoch, "flights": count,
                   **{key: value / count for key, value in totals.items()}}
            writer.writerow(row)
            stream.flush()
            print(f"epoch={epoch}/{epochs} " + " ".join(f"{key}={row[key]:.6f}" for key in totals), flush=True)
    state = representation_state(model)
    for name in REPRESENTATION_MODULES:
        if not any(not torch.equal(state[key], initial[key]) for key in state if key.startswith(name + ".")):
            raise RuntimeError(f"No parameter updates: {name}")
    if not all(torch.equal(initial_head[name], parameter) for name, parameter in model.hi_head.named_parameters()):
        raise RuntimeError("HI head changed during Stage 1")
    if not all(torch.isfinite(tensor).all().item() for tensor in state.values()):
        raise RuntimeError("Nonfinite representation weights")
    model.eval()
    probe_indices = indices[:3]
    probe = collate_fn([dataset[index] for index in probe_indices])
    with torch.no_grad():
        posterior = [value.cpu() for value in model.encode(probe["x"].to(device), probe["mask"].to(device))]
    if not all(torch.isfinite(value).all().item() for value in posterior):
        raise RuntimeError("Nonfinite posterior after final optimizer step")
    checkpoint = {"stage": "representation", "mode": mode, "epoch": epochs,
                  "config": config, "normalization": processed["normalization"], "columns": processed["columns"],
                  "state_dict": state, "training_indices": indices, "training_flights_total": len(dataset),
                  "device": str(device), "hi_head_unchanged": True,
                  "verification": {"indices": probe_indices, "posterior_before_save": posterior}}
    checkpoint.update({"structure": config["model"], "training_mode": "representation",
                       "loss_weights": representation_weights(config), "hi_minmax": None})
    path = run_dir / "representation.pt"
    torch.save(checkpoint, path)
    loaded, _ = load_representation(path, device)
    with torch.no_grad():
        after = loaded.encode(probe["x"].to(device), probe["mask"].to(device))
    for before, restored in zip(posterior, after):
        torch.testing.assert_close(before.to(device), restored, rtol=0, atol=0)
    print(f"Saved last epoch: {path}; updates/unchanged HI head/exact reload posterior verified", flush=True)
    return path


def validate_hi_config(config):
    validate_representation_config(config)
    expected = {"optimizer": "Adam", "learning_rate": 0.001, "weight_decay": 0.0,
                "batch_size_pairs": 128, "epochs": 100, "min_cycle_gap": 5, "kappa": 1.0,
                "l2_coefficient": 0.0001, "include_bias_in_l2": True,
                "sampling": "uniform_unit_then_uniform_valid_pair", "pairs_per_epoch": "N_train"}
    if config["training"]["hi"] != expected or config["smoke"]["hi_epochs"] != 3:
        raise ValueError("Stage 2 settings differ from approved v2 values")


def freeze_representation(model):
    for name in REPRESENTATION_MODULES:
        module = getattr(model, name)
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None


def extract_training_means(model, dataset, batch_size, device):
    means, units, cycles = [], [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    with torch.no_grad():
        for batch in loader:
            mu_i = model.encode(batch["x"].to(device), batch["mask"].to(device))[0]
            if not torch.isfinite(mu_i).all():
                raise RuntimeError("Nonfinite frozen training representation")
            means.append(mu_i)
            units.append(batch["unit"].to(device))
            cycles.append(batch["cycle"].to(device))
    if not means:
        raise ValueError("No training flights")
    return torch.cat(means), torch.cat(units), torch.cat(cycles)


def valid_pair_pools(units, cycles, min_cycle_gap):
    if units.ndim != 1 or units.shape != cycles.shape or units.device != cycles.device or min_cycle_gap <= 0:
        raise ValueError("Pair construction needs matching same-device identifiers and positive gap")
    pools = []
    for unit in torch.unique(units):
        indices = torch.where(units == unit)[0]
        indices = indices[torch.argsort(cycles[indices])]
        ordered_cycles = cycles[indices]
        if torch.unique(ordered_cycles).numel() != len(indices):
            raise ValueError(f"Duplicate unit/cycle keys for unit {unit.item()}")
        offsets = torch.nonzero(ordered_cycles[None, :] - ordered_cycles[:, None] >= min_cycle_gap)
        if len(offsets):
            pools.append((unit, indices[offsets]))
    if not pools:
        raise ValueError(f"No same-unit flight pairs with cycle gap >= {min_cycle_gap}")
    return pools


def sample_pairs(pools, count):
    if not pools or count <= 0:
        raise ValueError("Sampling needs nonempty valid pair pools and positive count")
    device = pools[0][1].device
    chosen_units = torch.randint(len(pools), (count,), device=device)
    pairs = torch.empty(count, 2, dtype=torch.int64, device=device)
    for index, (_, candidates) in enumerate(pools):
        slots = torch.where(chosen_units == index)[0]
        # Each candidate is one entire (early, late) pair; randint is uniform
        # over pairs, including when early flights have unequal candidate counts.
        draws = torch.randint(len(candidates), (len(slots),), device=device)
        pairs[slots] = candidates[draws]
    return pairs


def load_ocald(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    stage = checkpoint['stage']
    if checkpoint.get('score_mode', 'hi_head') != checkpoint['config']['model'].get('score_mode', 'hi_head'):
        raise ValueError('Checkpoint score_mode mismatch')
    if stage == 'hi':
        validate_hi_config(checkpoint['config'])
    elif stage == 'joint':
        validate_joint_config(checkpoint['config'])
        if checkpoint['loss_weights'] != checkpoint['config']['training']['joint']['loss_weights']:
            raise ValueError('Joint checkpoint weights differ from its configuration snapshot')
    elif stage in ('ocald_no_kli', 'ocald_s2'):
        validate_hi_config(checkpoint['config'])
    else:
        raise ValueError('Expected an HI or joint checkpoint')
    model = OCALD(checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(device).eval()
    freeze_representation(model)
    return model, checkpoint


def fit_score_scale(scores):
    if scores.ndim != 1 or not len(scores) or not torch.isfinite(scores).all():
        raise ValueError('Score scale needs finite dev predictions')
    return {'min': scores.min().item(), 'max': scores.max().item(),
            'count': len(scores), 'fit_split': 'dev', 'available': scores.max().item() != scores.min().item()}


def scale_score(scores, parameters):
    if parameters['fit_split'] != 'dev' or parameters['max'] == parameters['min']:
        raise ValueError('Dev score range is zero or invalid; score-scaled HI unavailable')
    return (parameters['max'] - scores) / (parameters['max'] - parameters['min'])


def scale_hi(hi_raw, parameters):
    if parameters["max"] == parameters["min"]:
        raise ValueError("Training HI range is zero; scaled HI is unavailable")
    return (hi_raw - parameters["min"]) / (parameters["max"] - parameters["min"])


def train_hi(config, representation_path, smoke):
    validate_hi_config(config)
    if config['model'].get('score_mode', 'hi_head') != 'hi_head':
        raise ValueError('Scalar I supports joint training only')
    seed = config["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, source = load_representation(representation_path, device)
    if any(source["config"][key] != config[key] for key in ("model", "data")):
        raise ValueError("Representation model/data settings differ from HI configuration")
    if not smoke and source["mode"] != "formal":
        raise ValueError("Formal HI training requires a formal Stage 1 checkpoint")
    processed = load_processed(config)
    for field in ("x", "w"):
        for key in ("min", "max"):
            if not torch.equal(source["normalization"][field][key], processed["normalization"][field][key]):
                raise ValueError("Representation normalization differs from prepared training data")
    if source["columns"] != processed["columns"]:
        raise ValueError("Representation column names differ from training data")
    freeze_representation(model)
    before = representation_state(model)  # Includes representation buffers.
    initial_head = {name: p.detach().cpu().clone() for name, p in model.hi_head.named_parameters()}
    dataset = FlightDataset(processed, "dev")
    means, units, cycles = extract_training_means(model, dataset, config["training"]["representation"]["batch_size"], device)
    settings = config["training"]["hi"]
    pools = valid_pair_pools(units, cycles, settings["min_cycle_gap"])
    mode = "smoke" if smoke else "formal"
    epochs = config["smoke"]["hi_epochs"] if smoke else settings["epochs"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(config["output"]["root"]) / f"hi_{mode}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    print(f"Stage=hi mode={mode} source={source['mode']} device={device} epochs={epochs} flights={len(dataset)} pairs_per_epoch={len(dataset)}", flush=True)
    print(f"Valid pairs per unit: {[(int(unit.item()), len(pairs)) for unit, pairs in pools]}; run_dir={run_dir}", flush=True)
    optimizer = torch.optim.Adam(model.hi_head.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    last_pairs = None
    with (run_dir / "losses.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["stage", "mode", "epoch", "pairs", "loss", "reconstruction", "kl_i", "kl_c", "order", "l2"])
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            model.hi_head.train()
            pairs = sample_pairs(pools, len(dataset))
            totals = {key: 0.0 for key in ("loss", "reconstruction", "kl_i", "kl_c", "order", "l2")}
            for start in range(0, len(pairs), settings["batch_size_pairs"]):
                pair = pairs[start:start + settings["batch_size_pairs"]]
                optimizer.zero_grad(set_to_none=True)
                losses = training_loss(model.hi_head, hi_weights(config),
                                       r_early=model.hi_head(means[pair[:, 0]]),
                                       r_late=model.hi_head(means[pair[:, 1]]), kappa=settings['kappa'])
                if not all(torch.isfinite(value).item() for value in losses.values()):
                    raise RuntimeError(f"Nonfinite HI loss at epoch {epoch}; no configuration changed")
                losses["loss"].backward()
                if not all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.hi_head.parameters()):
                    raise RuntimeError("Missing or nonfinite HI head gradients")
                optimizer.step()
                for key in totals:
                    totals[key] += losses[key].detach().item() * len(pair)
            row = {"stage": "hi", "mode": mode, "epoch": epoch, "pairs": len(pairs),
                   **{key: value / len(pairs) for key, value in totals.items()}}
            writer.writerow(row)
            stream.flush()
            print(f"epoch={epoch}/{epochs} " + " ".join(f"{key}={row[key]:.6f}" for key in totals), flush=True)
            last_pairs = pairs.detach().cpu()
    after = representation_state(model)
    if not all(torch.equal(before[key], after[key]) for key in before):
        raise RuntimeError("Frozen representation weights or buffers changed")
    if any(getattr(model, name).training or any(p.requires_grad or p.grad is not None for p in getattr(model, name).parameters()) for name in REPRESENTATION_MODULES):
        raise RuntimeError("Representation was not frozen/eval throughout HI training")
    if not any(not torch.equal(initial_head[name], p.detach().cpu()) for name, p in model.hi_head.named_parameters()):
        raise RuntimeError("HI head did not update")
    model.eval()
    with torch.no_grad():
        train_hi_raw = torch.sigmoid(-model.hi_head(means)).cpu()
    if not torch.isfinite(train_hi_raw).all() or not ((train_hi_raw > 0) & (train_hi_raw < 1)).all():
        raise RuntimeError("Training HI values are nonfinite or outside (0,1)")
    minmax = {"min": train_hi_raw.min().item(), "max": train_hi_raw.max().item(), "fit_split": "dev", "count": len(dataset)}
    if minmax["max"] == minmax["min"]:
        print("Training HI range is zero; scaled HI is unavailable", flush=True)
    probe = collate_fn([dataset[index] for index in range(min(3, len(dataset)))])
    with torch.no_grad():
        probe_hi = model.predict_hi(probe["x"].to(device), probe["mask"].to(device)).cpu()
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if not all(torch.isfinite(value).all().item() for value in state.values()):
        raise RuntimeError("Nonfinite HI checkpoint weights")
    checkpoint = {"stage": "hi", "mode": mode, "epoch": epochs, "config": config, "state_dict": state,
                  "normalization": source["normalization"], "columns": source["columns"], "hi_minmax": minmax,
                  "source_mode": source["mode"], "source_epoch": source["epoch"], "device": str(device),
                  "verification": {"representation_before": before, "initial_head": initial_head, "last_pairs": last_pairs,
                                   "train_hi_raw": train_hi_raw, "units": units.cpu(), "cycles": cycles.cpu(),
                                   "probe_hi_before_save": probe_hi}}
    checkpoint.update({"structure": config["model"], "training_mode": "hi", "loss_weights": hi_weights(config)})
    path = run_dir / "ocald.pt"
    torch.save(checkpoint, path)
    restored, _ = load_ocald(path, device)
    with torch.no_grad():
        reloaded_hi = restored.predict_hi(probe["x"].to(device), probe["mask"].to(device)).cpu()
    torch.testing.assert_close(probe_hi, reloaded_hi, rtol=0, atol=0)
    print(f"Training raw HI range={minmax['min']:.9f}..{minmax['max']:.9f}; saved last epoch: {path}", flush=True)
    print("Frozen state exact; head updated; independent HI reload exact", flush=True)
    return path


def state_hash(state):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in state.values())).hexdigest()

def datasets_for(processed, smoke):
    datasets = {}
    for split in ('dev', 'test'):
        dataset = FlightDataset(processed, split)
        expected = list(range(1, 10)) if split == 'dev' else list(range(10, 16))
        if sorted({f['unit'] for f in dataset.flights}) != expected:
            raise ValueError(f'Unexpected {split} unit set')
        keys = [(f['unit'], f['cycle']) for f in dataset.flights]
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate flight keys')
        if smoke:
            selected = []
            for unit in expected:
                flights = sorted((f for f in dataset.flights if f['unit'] == unit), key=lambda f: f['cycle'])
                indices = np.unique(np.linspace(0, len(flights) - 1, min(8, len(flights))).round().astype(int))
                selected.extend(flights[i] for i in indices)
            dataset.flights = selected
        datasets[split] = dataset
    return datasets

def move_batch(flights, device):
    return {key: value.to(device) for key, value in collate_fn(flights).items()}

def joint_schedule(flights, epochs, batches, seed):
    units = torch.tensor([f['unit'] for f in flights])
    cycles = torch.tensor([f['cycle'] for f in flights])
    groups = [torch.where(units == unit)[0] for unit in torch.unique(units)]
    if len(groups) < 4 or any(len(g) < 4 or cycles[g].max() - cycles[g].min() < 5 for g in groups):
        raise ValueError('Grouped sampling needs >=4 units with >=4 flights and valid cycle gaps')
    flight_rng = torch.Generator().manual_seed(seed)
    batch_list, pair_list = [], []
    # Existing sampler uses global CPU RNG; isolate it from initialization/noise.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        for _ in range(epochs * batches):
            selections = []
            for group_index in torch.randperm(len(groups), generator=flight_rng)[:4]:
                group = groups[int(group_index)]
                while True:
                    selected = group[torch.randperm(len(group), generator=flight_rng)[:4]]
                    if cycles[selected].max() - cycles[selected].min() >= 5:
                        selections.append(selected)
                        break
            indices = torch.cat(selections)
            u, c = units[indices], cycles[indices]
            assert len(torch.unique(indices)) == 16
            assert torch.unique(u, return_counts=True)[1].tolist() == [4] * 4
            pairs = sample_pairs(valid_pair_pools(u, c, 5), 16)
            assert (u[pairs[:, 0]] == u[pairs[:, 1]]).all()
            assert (c[pairs[:, 1]] - c[pairs[:, 0]] >= 5).all()
            batch_list.append(indices)
            pair_list.append(pairs)
    return {'flights': torch.stack(batch_list).reshape(epochs, batches, 16),
            'pairs': torch.stack(pair_list).reshape(epochs, batches, 16, 2)}

def encoder_order_norm(model, order):
    params = list(model.sensor_encoder.parameters())
    grads = torch.autograd.grad(order, params, retain_graph=True, allow_unused=True)
    # Variance/C output heads are not on the ranking path; GRU and mu_I are.
    names = list(model.sensor_encoder.named_parameters())
    for (name, _), grad in zip(names, grads):
        if name.startswith(('gru.', 'mu_i.')) and grad is None:
            raise RuntimeError(f'Missing ranking gradient: {name}')
    valid = [g for g in grads if g is not None]
    if not valid or not all(torch.isfinite(g).all() for g in valid):
        raise RuntimeError('Invalid encoder ranking gradients')
    return torch.stack([g.detach().square().sum() for g in valid]).sum().sqrt().item()

def check_mechanisms(config, batch, pairs):
    model = OCALD(config).to(batch['x'].device)
    model.train()  # cuDNN GRU backward requires a training-mode forward.
    original_mask = batch['mask'].clone()
    output = model(batch['x'], batch['w'], batch['mask'])
    r = model.score(output['mu_i'])
    order = order_loss(r[pairs[:, 0]], r[pairs[:, 1]], 1)
    if model.score_mode == 'scalar_i':
        assert model.hi_head is None and output['mu_i'].shape == (len(r), 1)
        torch.testing.assert_close(r, output['mu_i'].squeeze(-1), rtol=0, atol=0)
    norm = encoder_order_norm(model, order)
    if norm <= 0:
        raise RuntimeError('Ranking gradient did not reach encoder')
    other = list(model.decoder.parameters()) + list(model.conditional_prior.parameters())
    assert all(g is None for g in torch.autograd.grad(order, other, allow_unused=True, retain_graph=True))
    losses = representation_loss(batch['x'], batch['mask'], output, 1)
    torch.testing.assert_close(losses['loss'], losses['reconstruction'] + losses['kl_i'] + losses['kl_c'])
    for term, module in (('kl_i', model.sensor_encoder.mu_i), ('kl_c', model.conditional_prior)):
        grads = torch.autograd.grad(losses[term], tuple(module.parameters()), retain_graph=True)
        assert all(torch.isfinite(g).all() for g in grads) and sum(g.square().sum().item() for g in grads) > 0
    weighted = training_loss(model.hi_head, config['training']['joint']['loss_weights'],
        x=batch['x'], mask=batch['mask'], output=output, r_early=r[pairs[:, 0]], r_late=r[pairs[:, 1]])
    weights = config['training']['joint']['loss_weights']
    expected = sum(weights[key] * weighted[key] for key in weights)
    torch.testing.assert_close(weighted['loss'], expected, rtol=1e-6, atol=1e-5)
    if model.score_mode == 'scalar_i':
        # Compare gradients, since the legacy flight-summed objective computes
        # its sum before means and the logged KL means are sibling tensors.
        params = tuple(model.sensor_encoder.parameters()) + tuple(model.conditional_prior.parameters())
        actual_grads = torch.autograd.grad(weighted['loss'], params, retain_graph=True)
        expected_grads = torch.autograd.grad(expected, params, retain_graph=True)
        for actual, reference in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
    before = reconstruction_loss(batch['x'], output['x_hat'], batch['mask'], 1)
    changed_x = batch['x'].masked_fill(~batch['mask'][:, :, None], float('nan'))
    changed_prediction = output['x_hat'].masked_fill(~batch['mask'][:, :, None], float('nan'))
    torch.testing.assert_close(before, reconstruction_loss(changed_x, changed_prediction, batch['mask'], 1), rtol=0, atol=0)
    model.eval()
    with torch.no_grad():
        posterior = model.encode(batch['x'], batch['mask'])
        for a, b in zip(posterior, model.encode(changed_x, batch['mask'])):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        extended_mask = F.pad(batch['mask'], (0, 8), value=False)
        extended_x = F.pad(batch['x'], (0, 0, 0, 8), value=123)
        for a, b in zip(posterior, model.encode(extended_x, extended_mask)):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
        mu_i, _, mu_c, _ = posterior
        decoder_w = batch['w'].masked_fill(~batch['mask'][:, :, None], 0)
        baseline = model.decode(mu_i, mu_c, time_positions(batch['mask']), decoder_w)
        extended_w = F.pad(decoder_w, (0, 0, 0, 8), value=0)
        extended = model.decode(mu_i, mu_c, time_positions(extended_mask), extended_w)
        torch.testing.assert_close(baseline[batch['mask']], extended[extended_mask], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(model.predict_score(batch['x'], batch['mask']), model.predict_score(extended_x, extended_mask), rtol=1e-6, atol=1e-6)
        changed_w = decoder_w + batch['mask'][:, :, None] * .25
        changed = model.decode(mu_i, mu_c, time_positions(batch['mask']), changed_w)
        if model.decoder_use_w:
            assert not torch.equal(baseline[batch['mask']], changed[batch['mask']])
        else:
            torch.testing.assert_close(baseline, changed, rtol=0, atol=0)
    devices = [batch['x'].device.index] if batch['x'].is_cuda else []
    model.train()
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(42)
        original = model(batch['x'], batch['w'], batch['mask'])
        torch.manual_seed(42)
        changed = model(batch['x'], batch['w'] + .25, batch['mask'])
    for key in ('mu_i', 'logvar_i', 'mu_c', 'logvar_c'):
        torch.testing.assert_close(original[key], changed[key], rtol=0, atol=0)
    torch.testing.assert_close(model.score(original['mu_i']),
                               model.score(changed['mu_i']), rtol=0, atol=0)
    w_grad_norm = None
    if model.decoder_use_w:
        w_grad = torch.autograd.grad(losses['reconstruction'], model.decoder.layers[0].weight, retain_graph=True)[0][:, config['model']['latent_i_dim'] + config['model']['latent_c_dim']:][:, :config['data']['condition_dim']]
        w_grad_norm = w_grad.double().norm().item()
        assert torch.isfinite(w_grad).all() and w_grad_norm > 0
    assert torch.equal(batch['mask'], original_mask)
    assert (batch['unit'][pairs[:, 0]] == batch['unit'][pairs[:, 1]]).all()
    assert (batch['cycle'][pairs[:, 1]] - batch['cycle'][pairs[:, 0]] >= 5).all()
    assert order_loss(torch.tensor([0.]), torch.tensor([1.]), 1) < order_loss(torch.tensor([1.]), torch.tensor([0.]), 1)
    print(f'PASS: OCALD pair/mask/base-loss/path checks; encoder ranking gradient norm={norm:.9g}', flush=True)
    return {'initial_encoder_order_gradient_norm': norm, 'mask_padding_checked': True,
            'order_only_decoder_prior_gradients_absent': True, 'pair_direction_checked': True,
            'decoder_use_w': model.decoder_use_w, 'w_independent_encode_hi_checked': True,
            'decoder_w_input_gradient_norm': w_grad_norm, 'score_mode': model.score_mode,
            'both_kl_gradients_and_unit_weights_checked': True,
            'scalar_same_forward_no_detach_no_head_checked': model.score_mode == 'scalar_i'}


def representation_weights(config):
    settings = config['training']['representation']
    return {'reconstruction': 1., 'kl_i': settings['kl_i_weight'], 'kl_c': settings['kl_c_weight'], 'order': 0., 'l2': 0.}


def hi_weights(config):
    return {'reconstruction': 0., 'kl_i': 0., 'kl_c': 0., 'order': 1., 'l2': config['training']['hi']['l2_coefficient']}


def validate_joint_config(config):
    validate_hi_config(config)
    if 'joint' not in config['training']:
        raise ValueError('Joint training requires a training.joint configuration')
    settings = config['training']['joint']
    if config['model'].get('score_mode', 'hi_head') == 'scalar_i':
        if not config['model'].get('decoder_use_w') or settings['loss_weights'] != dict(reconstruction=1., kl_i=1., kl_c=1., order=10., l2=0.):
            raise ValueError('Scalar I requires W decoder and approved loss weights')
    expected = {'optimizer': 'Adam', 'learning_rate': .001, 'weight_decay': 0., 'batch_size': 16,
                'epochs': 100, 'sigma_x_squared': 1., 'gradient_clip_norm': 5., 'kappa': 1.,
                'min_cycle_gap': 5, 'pairs_per_batch': 16, 'batch_scheme': 'grouped',
                'units_per_batch': 4, 'flights_per_unit': 4}
    if {k: v for k, v in settings.items() if k != 'loss_weights'} != expected:
        raise ValueError('Joint training settings differ from the approved grouped pipeline')
    weights = settings['loss_weights']
    if set(weights) != {'reconstruction', 'kl_i', 'kl_c', 'order', 'l2'}:
        raise ValueError('Missing joint loss weights')
    if any(not np.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError('Loss weights must be finite and nonnegative')
    if config['smoke']['joint_epochs'] != 3 or config['smoke']['joint_batches_per_epoch'] != 2 or config['smoke']['joint_flights_per_unit'] != 8:
        raise ValueError('Joint smoke budget differs from the approved settings')


def train_joint(config, smoke):
    validate_joint_config(config)
    settings = config['training']['joint']
    weights = settings['loss_weights']
    seed = config['training']['seed']
    processed = load_processed(config)
    datasets = datasets_for(processed, smoke)
    epochs = config['smoke']['joint_epochs'] if smoke else settings['epochs']
    batches = config['smoke']['joint_batches_per_epoch'] if smoke else math.ceil(len(datasets['dev'])/settings['batch_size'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        torch.set_num_threads(1)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root = Path(config['output']['root'])/f'joint_{"smoke" if smoke else "full"}_{stamp}'
    root.mkdir(parents=True, exist_ok=False)
    (root/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    schedule = joint_schedule(datasets['dev'].flights, epochs, batches, seed)
    torch.save(schedule, root/'sampling_schedule.pt')
    batch = move_batch([datasets['dev'].flights[i] for i in schedule['flights'][0,0].tolist()], device)
    checks = check_mechanisms(config, batch, schedule['pairs'][0,0].to(device))
    # Restore the original joint initialization/noise rule after mechanism checks.
    torch.manual_seed(seed)
    model = OCALD(config).to(device)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    logs = []
    print(f'Stage=joint device={device} epochs={epochs} weights={weights} run_dir={root}', flush=True)
    for epoch in range(epochs):
        model.train()
        totals = dict.fromkeys(('loss','reconstruction','kl_i','kl_c','order','l2'), 0.)
        clips = []
        order_norm = None
        for step, (indices, cpu_pairs) in enumerate(zip(schedule['flights'][epoch],schedule['pairs'][epoch])):
            batch = move_batch([datasets['dev'].flights[i] for i in indices.tolist()],device)
            pairs = cpu_pairs.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch['x'],batch['w'],batch['mask'])
            r = model.score(output['mu_i'])
            terms = training_loss(model.hi_head,weights,x=batch['x'],mask=batch['mask'],output=output,
                                  r_early=r[pairs[:,0]],r_late=r[pairs[:,1]],
                                  sigma_x_squared=settings['sigma_x_squared'],kappa=settings['kappa'])
            if not all(torch.isfinite(v) for v in terms.values()):
                raise RuntimeError('Nonfinite joint loss; no silent numerical fallback')
            if step == 0:
                order_norm = encoder_order_norm(model,terms['order'])
                if epoch == 0 and order_norm <= 0:
                    raise RuntimeError('Ranking gradient failed to reach GRU/mu_I')
            terms['loss'].backward()
            clips.append(torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip_norm'],error_if_nonfinite=True).item())
            optimizer.step()
            for key in totals:
                totals[key] += terms[key].detach().item()
        logs.append({'epoch':epoch+1,**{k:v/batches for k,v in totals.items()},
                     'encoder_order_grad_norm':order_norm,'mean_preclip_grad_norm':float(np.mean(clips))})
        pd.DataFrame(logs).to_csv(root/'losses.csv',index=False,float_format='%.17g')
        if epoch == 0 or (epoch+1)%10 == 0 or epoch+1 == epochs:
            print(f'epoch={epoch+1}/{epochs}: '+ ' '.join(f'{k}={logs[-1][k]:.6f}' for k in totals),flush=True)
    rng = torch.cuda.get_rng_state(device) if device.type == 'cuda' else torch.get_rng_state()
    digest = hashlib.sha256(rng.numpy().tobytes()).hexdigest()
    model.eval()
    raw, scores = {}, {}
    for split,dataset in datasets.items():
        means,_,_ = extract_training_means(model,dataset,16,device)
        with torch.no_grad():
            device_scores = model.score(means)
            scores[split] = device_scores.cpu()
            raw[split] = scores[split] if model.score_mode == 'scalar_i' else torch.sigmoid(-device_scores).cpu()
        if not torch.isfinite(raw[split]).all():
            raise RuntimeError('Nonfinite final raw HI')
    scale = {'min':raw['dev'].min().item(),'max':raw['dev'].max().item(),'count':len(raw['dev']),'fit_split':'dev'}
    score_scale = fit_score_scale(scores['dev'])
    saved = {'score_mode': model.score_mode, 'scores': scores, 'score_minmax': score_scale,
             'stage':'joint','training_mode':'joint','structure':{**config['model'], 'decoder_use_w':model.decoder_use_w, 'score_mode':model.score_mode},
             'mode':'smoke' if smoke else 'full','epoch':epochs,'seed':seed,'config':config,
             'loss_weights':weights,'actual_training':{**settings,'epochs':epochs,'seed':seed},
             'normalization':processed['normalization'],'columns':processed['columns'],
             'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
             'initial_state_dict':initial,'initial_sha256':state_hash(initial),
             'initialization_rule':'pytorch_defaults_scalar_no_head_then_isolated_w_layer' if model.score_mode == 'scalar_i' else 'legacy_modules_then_isolated_decoder_input_layer' if model.decoder_use_w else 'legacy_default',
             'sampling_rule':'CPU seed42 flight generator; independent CPU seed42 uniform-unit/valid-pair RNG; saved schedule',
             'reparameterization_rule':'torch seed42 after mechanism checks; default initialization then per-forward randn_like for I and C; architecture-dependent RNG consumption',
             'raw_hi_kind':'score' if model.score_mode == 'scalar_i' else 'sigmoid_hi',
             'reparameterization_rng_digest':digest,'raw_hi':raw,'hi_minmax':scale,
             'flight_keys':{s:[[f['unit'],f['cycle']] for f in d.flights] for s,d in datasets.items()}}
    torch.save(saved,root/'checkpoint.pt')
    (root/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    restored,_ = load_ocald(root/'checkpoint.pt',device)
    for split,dataset in datasets.items():
        means,_,_ = extract_training_means(restored,dataset,16,device)
        with torch.no_grad():
            device_score = restored.score(means)
            score = device_score.cpu()
            torch.testing.assert_close(score, scores[split], rtol=0, atol=0)
            torch.testing.assert_close(score if restored.score_mode == 'scalar_i' else torch.sigmoid(-device_score).cpu(), raw[split], rtol=0, atol=0)
    print(f'PASS: joint last checkpoint exact reload; checkpoint={root/"checkpoint.pt"}',flush=True)
    return root/'checkpoint.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--pipeline',choices=['two_stage','joint'])
    parser.add_argument('--stage',choices=['representation','hi','joint'],help='Optional individual-stage entry (legacy compatible)')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--representation')
    args = parser.parse_args()
    if args.pipeline and args.stage:
        parser.error('Choose --pipeline or --stage, not both')
    config = load_config(args.config)
    if not torch.cuda.is_available():
        torch.set_num_threads(1)
    flow = args.stage or args.pipeline or config['training'].get('pipeline','two_stage')
    if flow not in ('representation', 'hi', 'joint', 'two_stage'):
        parser.error('Unknown configured pipeline')
    if flow == 'representation':
        path = train_representation(config,args.smoke)
    elif flow == 'hi':
        if args.representation is None:
            parser.error('hi requires --representation')
        path = train_hi(config,args.representation,args.smoke)
    elif flow == 'joint':
        path = train_joint(config,args.smoke)
    else:
        representation = train_representation(config,args.smoke)
        path = train_hi(config,representation,args.smoke)
    print(f'Pipeline complete; checkpoint={path}',flush=True)


if __name__ == '__main__':
    main()
