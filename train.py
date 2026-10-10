"""Approved two-stage OCALD training and independent checkpoint loading."""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from data import FlightDataset, collate_fn, load_config, load_processed
from losses import hi_loss, representation_loss
from model import OCALD


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
        fields = ["stage", "mode", "epoch", "flights", "loss", "reconstruction", "kl_i", "kl_c"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            model.train()
            totals = {key: 0.0 for key in ("loss", "reconstruction", "kl_i", "kl_c")}
            count = 0
            for batch in loader:
                x, w, mask = (batch[key].to(device) for key in ("x", "w", "mask"))
                optimizer.zero_grad(set_to_none=True)
                losses = representation_loss(x, mask, model(x, w, mask), settings["sigma_x_squared"])
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
    if checkpoint["stage"] != "hi":
        raise ValueError("Expected an HI checkpoint")
    validate_hi_config(checkpoint["config"])
    model = OCALD(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    freeze_representation(model)
    return model, checkpoint


def scale_hi(hi_raw, parameters):
    if parameters["max"] == parameters["min"]:
        raise ValueError("Training HI range is zero; scaled HI is unavailable")
    return (hi_raw - parameters["min"]) / (parameters["max"] - parameters["min"])


def train_hi(config, representation_path, smoke):
    validate_hi_config(config)
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
        writer = csv.DictWriter(stream, fieldnames=["stage", "mode", "epoch", "pairs", "loss", "order", "l2"])
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            model.hi_head.train()
            pairs = sample_pairs(pools, len(dataset))
            totals = {key: 0.0 for key in ("loss", "order", "l2")}
            for start in range(0, len(pairs), settings["batch_size_pairs"]):
                pair = pairs[start:start + settings["batch_size_pairs"]]
                optimizer.zero_grad(set_to_none=True)
                losses = hi_loss(model.hi_head(means[pair[:, 0]]), model.hi_head(means[pair[:, 1]]),
                                 model.hi_head, settings["kappa"], settings["l2_coefficient"])
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
    path = run_dir / "ocald.pt"
    torch.save(checkpoint, path)
    restored, _ = load_ocald(path, device)
    with torch.no_grad():
        reloaded_hi = restored.predict_hi(probe["x"].to(device), probe["mask"].to(device)).cpu()
    torch.testing.assert_close(probe_hi, reloaded_hi, rtol=0, atol=0)
    print(f"Training raw HI range={minmax['min']:.9f}..{minmax['max']:.9f}; saved last epoch: {path}", flush=True)
    print("Frozen state exact; head updated; independent HI reload exact", flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=["representation", "hi"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--representation")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage == "representation":
        train_representation(config, args.smoke)
    else:
        if args.representation is None:
            parser.error("hi requires --representation")
        train_hi(config, args.representation, args.smoke)


if __name__ == "__main__":
    main()
