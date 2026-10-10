"""Stage 1 representation training; HI training is a later task."""

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
from losses import representation_loss
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=["representation"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    train_representation(load_config(args.config), args.smoke)


if __name__ == "__main__":
    main()
