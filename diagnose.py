"""D1 scale / D2 blocks: paired HI heads on frozen posterior means."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from data import FlightDataset, collate_fn, load_config, load_processed, read_reference
from losses import hi_loss, order_loss
from metrics import mae, spearman
from model import HIHead
from train import (extract_training_means, freeze_representation,
                   load_representation, representation_state, sample_pairs,
                   valid_pair_pools, validate_hi_config)


def fit_transform(dev):
    values = dev.double()
    mean, std = values.mean(0), values.std(0, unbiased=False)
    return {"mean": mean.cpu(), "std": std.cpu(),
            "denominator": torch.where(std == 0, torch.ones_like(std), std).cpu(),
            "zero_std_dimensions": torch.where(std == 0)[0].cpu().tolist(),
            "fit_split": "dev", "count": len(dev), "ddof": 0}


def apply_transform(values, parameters):
    if parameters is None:
        return values
    return ((values.double() - parameters["mean"].to(values.device))
            / parameters["denominator"].to(values.device)).to(values.dtype)


def extract_blocks(model, dataset, batch_size, device):
    """Take I/C from the SAME encode call and check every stored flight key."""
    blocks, units, cycles = {"I": [], "C": []}, [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    with torch.no_grad():
        for batch in loader:
            mu_i, _, mu_c, _ = model.encode(batch["x"].to(device), batch["mask"].to(device))
            for block, values in (("I", mu_i), ("C", mu_c)):
                assert values.shape == (len(batch["unit"]), 8) and torch.isfinite(values).all()
                blocks[block].append(values)
            units.append(batch["unit"].to(device))
            cycles.append(batch["cycle"].to(device))
    units, cycles = torch.cat(units), torch.cat(cycles)
    assert units.cpu().tolist() == [flight["unit"] for flight in dataset.flights]
    assert cycles.cpu().tolist() == [flight["cycle"] for flight in dataset.flights]
    return {block: (torch.cat(values), units, cycles) for block, values in blocks.items()}


def pair_schedule(pools, count, epochs, seed):
    """Reuse the existing sampler inside a separate, locally seeded RNG state."""
    device = pools[0][1].device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.random.fork_rng(devices=devices):
        if device.type == "cuda":
            torch.cuda.set_rng_state(generator.get_state(), device)
        else:
            torch.set_rng_state(generator.get_state())
        return torch.stack([sample_pairs(pools, count) for _ in range(epochs)])


def latent_statistics(data, transform):
    rows = []
    for split, (values, units, _) in data.items():
        groups = [("all", values)]
        if split == "dev":
            groups += [(str(int(unit)), values[units == unit]) for unit in torch.unique(units)]
        for unit, group in groups:
            group = group.double()
            mean, std, lo, hi = group.mean(0), group.std(0, unbiased=False), group.min(0).values, group.max(0).values
            for dim in range(group.shape[1]):
                rows.append({"split": split, "unit": unit, "dimension": dim, "count": len(group),
                             "mean": mean[dim].item(), "std": std[dim].item(),
                             "min": lo[dim].item(), "max": hi[dim].item(),
                             "dev_fit_mean": transform["mean"][dim].item(),
                             "dev_fit_std": transform["std"][dim].item(),
                             "denominator": transform["denominator"][dim].item(),
                             "zero_dev_std": dim in transform["zero_std_dimensions"]})
    return pd.DataFrame(rows)


def ranking_rows(name, split, scores, units, cycles, reference, gap, kappa):
    """Enumerate every valid pair; overall ranking metrics weight units equally."""
    hi = torch.sigmoid(-scores)
    rows = []
    for unit, pairs in valid_pair_pools(units, cycles, gap):
        indices = torch.where(units == unit)[0]
        indices = indices[torch.argsort(cycles[indices])]
        # Independently check enumeration rather than estimate by sampling.
        ordered_indices = indices.cpu().tolist()
        cycle_values = cycles.cpu().tolist()
        expected = {(a, b) for a in ordered_indices for b in ordered_indices
                    if cycle_values[b] - cycle_values[a] >= gap}
        assert set(map(tuple, pairs.cpu().tolist())) == expected
        early, late = scores[pairs[:, 0]], scores[pairs[:, 1]]
        values = hi[indices].cpu().numpy()
        rows.append({"experiment": name, "split": split, "unit": str(int(unit)),
                     "flights": len(indices), "pairs": len(pairs),
                     "correct_count": int((late > early).sum()), "tie_count": int((late == early).sum()),
                     "accuracy": (late > early).double().mean().item(),
                     "tie_rate": (late == early).double().mean().item(),
                     "order": order_loss(early, late, kappa).item(),
                     "spearman_hi_cycle": spearman(values, cycles[indices].cpu().numpy()),
                     "hi_min": float(values.min()), "hi_max": float(values.max()),
                     "hi_std": float(values.astype(np.float64).std(ddof=0)),
                     "mae": mae(values, reference[indices.cpu().numpy()])})
    if len(rows) != torch.unique(units).numel():
        raise ValueError("A unit has no valid pairs; cannot form a full unit-equal comparison")
    frame = pd.DataFrame(rows)
    values = hi.cpu().numpy()
    overall = {"experiment": name, "split": split, "unit": "all", "flights": len(scores),
               **{key: int(frame[key].sum()) for key in ("pairs", "correct_count", "tie_count")},
               **{key: float(frame[key].mean()) for key in ("accuracy", "tie_rate", "order", "spearman_hi_cycle")},
               "hi_min": float(values.min()), "hi_max": float(values.max()),
               "hi_std": float(values.astype(np.float64).std(ddof=0)), "mae": mae(values, reference)}
    return rows + [overall]


def plot_results(run_dir, logs, predictions, mode, diagnostic):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for name, table in logs.items():
        axes[0].plot(table.epoch, table.order, label=name)
        axes[1].plot(table.epoch, table.weighted_l2, label=name)
    for axis, title in zip(axes, ("Order loss", "Weighted L2")):
        axis.set(title=title, xlabel="Epoch", ylabel="Loss")
        axis.grid(alpha=.25)
        axis.legend()
    figure.suptitle(f"{diagnostic} — {mode}")
    figure.savefig(run_dir / "training_comparison.png", dpi=140)
    plt.close(figure)

    first = next(iter(predictions.values()))
    groups = list(first.groupby(["split", "unit"], sort=True))
    figure, axes = plt.subplots(len(groups), 1, figsize=(11, 2.2 * len(groups)), squeeze=False, layout="constrained")
    for axis, ((split, unit), reference) in zip(axes[:, 0], groups):
        reference = reference.sort_values("cycle")
        axis.plot(reference.cycle, reference.hi_gt, color="black", label="Reference HI")
        for name, table in predictions.items():
            group = table.loc[(table.split == split) & (table.unit == unit)].sort_values("cycle")
            line, = axis.plot(group.cycle, group.hi_raw, label=name)
            assert np.array_equal(line.get_ydata(), group.hi_raw.to_numpy())
            assert np.array_equal(line.get_xdata(), group.cycle.to_numpy())
        axis.set(title=f"{split} unit {unit}", xlabel="Flight cycle", ylabel="Raw HI")
        axis.legend(fontsize=8)
        axis.grid(alpha=.25)
    figure.suptitle(f"{diagnostic} raw HI comparison — {mode}")
    figure.savefig(run_dir / "hi_comparison.png", dpi=140)
    plt.close(figure)


def run(config, representation_path, smoke, experiment="scale"):
    validate_hi_config(config)
    diagnostic = "D1 scale" if experiment == "scale" else "D2 blocks"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(1)
    source_hash = hashlib.sha256(Path(representation_path).read_bytes()).hexdigest()
    model, source = load_representation(representation_path, device)
    if source["mode"] != "formal" or source["epoch"] != config["training"]["representation"]["epochs"]:
        raise ValueError("Diagnostics require a completed formal representation checkpoint, including smoke")
    if source["config"] != config:
        raise ValueError("Diagnostic config differs from representation checkpoint")
    processed = load_processed(config)
    assert source["columns"] == processed["columns"]
    for field in ("x", "w"):
        for key in ("min", "max"):
            assert torch.equal(source["normalization"][field][key], processed["normalization"][field][key])
    freeze_representation(model)
    before = representation_state(model)
    batch_size = config["training"]["representation"]["batch_size"]
    if experiment == "scale":
        data = {split: extract_training_means(model, FlightDataset(processed, split), batch_size, device)
                for split in ("dev", "test")}
        block_data = {"I": data}
    else:
        extracted = {split: extract_blocks(model, FlightDataset(processed, split), batch_size, device)
                     for split in ("dev", "test")}
        block_data = {block: {split: extracted[split][block] for split in extracted} for block in ("I", "C")}
        data = block_data["I"]
        # An independent pass and an independently collated probe verify the
        # reused I extraction and C output index against the actual flight keys.
        for split in ("dev", "test"):
            dataset = FlightDataset(processed, split)
            check_i, check_units, check_cycles = extract_training_means(model, dataset, batch_size, device)
            torch.testing.assert_close(data[split][0], check_i, rtol=0, atol=0)
            assert torch.equal(data[split][1], check_units) and torch.equal(data[split][2], check_cycles)
            indices = [0, len(dataset) // 2, len(dataset) - 1]
            probe = collate_fn([dataset[index] for index in indices])
            with torch.no_grad():
                posterior = model.encode(probe["x"].to(device), probe["mask"].to(device))
            for block, output_index in (("I", 0), ("C", 2)):
                torch.testing.assert_close(block_data[block][split][0][indices], posterior[output_index], rtol=1e-6, atol=1e-6)
    dev, units, cycles = data["dev"]
    block_transforms = {block: fit_transform(values["dev"][0]) for block, values in block_data.items()}
    for block, transform in block_transforms.items():
        # Independent NumPy fit confirms ddof=0 and all-dev statistics.
        array = block_data[block]["dev"][0].cpu().numpy().astype(np.float64)
        np.testing.assert_allclose(transform["mean"], array.mean(0), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(transform["std"], array.std(0, ddof=0), rtol=1e-12, atol=1e-12)
    toy = fit_transform(torch.tensor([[2., 1.], [2., 3.]]))
    assert toy["zero_std_dimensions"] == [0] and toy["denominator"][0] == 1
    torch.testing.assert_close(apply_transform(torch.tensor([[2., 5.]]), toy), torch.tensor([[0., 3.]]))
    mode = "smoke" if smoke else "formal"
    epochs = config["smoke"]["hi_epochs"] if smoke else config["training"]["hi"]["epochs"]
    run_dir = Path(config["output"]["root"]) / f"diagnostics_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    statistics = [latent_statistics(values, block_transforms[block]).assign(block=block) for block, values in block_data.items()]
    pd.concat(statistics, ignore_index=True).to_csv(run_dir / "latent_stats.csv", index=False, float_format="%.17g")
    print(f"{diagnostic} {mode}: device={device}, dev={len(dev)}, test={len(data['test'][0])}, epochs={epochs}; output={run_dir}", flush=True)
    zero_dimensions = {block: transform["zero_std_dimensions"] for block, transform in block_transforms.items()}
    print(f"Latent statistics saved BEFORE head training; zero std dimensions={zero_dimensions}", flush=True)

    settings, seed = config["training"]["hi"], config["training"]["seed"]
    pools = valid_pair_pools(units, cycles, settings["min_cycle_gap"])
    schedule = pair_schedule(pools, len(dev), epochs, seed)
    # Consume unrelated initialization randomness, then confirm sampler isolation.
    torch.manual_seed(seed)
    initial = HIHead(8, 16).to(device)
    assert torch.equal(schedule, pair_schedule(pools, len(dev), epochs, seed))
    if experiment == "scale":
        names = ("zi_raw", "zi_standardized")
        features = {name: data for name in names}
        transforms = {"zi_raw": None, "zi_standardized": block_transforms["I"]}
    else:
        names = ("zi_standardized", "zc_standardized", "zic_standardized")
        combined = {split: (torch.cat([block_data[block][split][0] for block in ("I", "C")], dim=1), u, c)
                    for split, (_, u, c) in data.items()}
        features = {names[0]: data, names[1]: block_data["C"], names[2]: combined}
        combined_transform = fit_transform(combined["dev"][0])
        for key in ("mean", "std", "denominator"):
            separate_parameters = torch.cat([block_transforms[b][key] for b in ("I", "C")])
            # GPU reductions can differ at float64 roundoff with 8 vs 16
            # columns. Use the same per-dimension fit in both input paths.
            torch.testing.assert_close(combined_transform[key], separate_parameters, rtol=1e-12, atol=1e-12)
            combined_transform[key] = separate_parameters
        combined_transform["zero_std_dimensions"] = torch.where(combined_transform["std"] == 0)[0].tolist()
        transforms = {names[0]: block_transforms["I"], names[1]: block_transforms["C"], names[2]: combined_transform}
        for split in ("dev", "test"):
            separate = torch.cat([apply_transform(block_data[b][split][0], block_transforms[b]) for b in ("I", "C")], dim=1)
            torch.testing.assert_close(separate, apply_transform(combined[split][0], combined_transform), rtol=0, atol=0)
    heads = {name: copy.deepcopy(initial) for name in names[:2]}
    if experiment == "blocks":
        # Independent seed reset makes the different-size head reproducible;
        # it does not imply matched capacity or matched first-layer weights.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            heads[names[2]] = HIHead(16, 16).to(device)
    initial_states = {name: {key: value.detach().cpu().clone() for key, value in head.state_dict().items()} for name, head in heads.items()}
    assert all(torch.equal(initial_states[names[0]][key], initial_states[names[1]][key]) for key in initial_states[names[0]])
    if experiment == "blocks":
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            repeat_initial = HIHead(16, 16)
        assert all(torch.equal(value, repeat_initial.state_dict()[key]) for key, value in initial_states[names[2]].items())
    parameter_counts = {name: sum(p.numel() for p in head.parameters()) for name, head in heads.items()}
    print(f"Head parameter counts: {parameter_counts}", flush=True)
    inputs = {name: {split: apply_transform(values, transforms[name]) for split, (values, _, _) in features[name].items()} for name in names}
    assert all(torch.isfinite(values).all() for splits in inputs.values() for values in splits.values())
    optimizers = {name: torch.optim.Adam(head.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
                  for name, head in heads.items()}
    rows = {name: [] for name in names}
    digests = {name: hashlib.sha256() for name in names}
    epoch_checks = []
    torch.save({"pairs": schedule.cpu(), "units": units.cpu(), "cycles": cycles.cpu(), "seed": seed}, run_dir / "pair_schedule.pt")
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    for epoch, pairs in enumerate(schedule, 1):
        epoch_digests = {name: hashlib.sha256() for name in names}
        totals = {name: {key: 0.0 for key in ("loss", "order", "l2")} for name in names}
        for start in range(0, len(pairs), settings["batch_size_pairs"]):
            batch = pairs[start:start + settings["batch_size_pairs"]]
            for name in names:
                head, optimizer, values = heads[name], optimizers[name], inputs[name]["dev"]
                head.train()
                optimizer.zero_grad(set_to_none=True)
                losses = hi_loss(head(values[batch[:, 0]]), head(values[batch[:, 1]]), head,
                                 settings["kappa"], settings["l2_coefficient"])
                if not all(torch.isfinite(value) for value in losses.values()):
                    raise RuntimeError(f"Nonfinite {diagnostic} loss: {name}, epoch={epoch}")
                losses["loss"].backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
                optimizer.step()
                digests[name].update(batch.cpu().numpy().tobytes())
                epoch_digests[name].update(batch.cpu().numpy().tobytes())
                for key in totals[name]:
                    totals[name][key] += losses[key].item() * len(batch)
        for name in names:
            row = {"mode": mode, "epoch": epoch, "pairs": len(pairs), **{key: value / len(pairs) for key, value in totals[name].items()}}
            row["weighted_l2"] = row["l2"] * settings["l2_coefficient"]
            rows[name].append(row)
        hashes = {name: digest.hexdigest() for name, digest in epoch_digests.items()}
        assert len(set(hashes.values())) == 1
        epoch_checks.append({"epoch": epoch, "identical": True, "sha256_by_experiment": hashes})
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"epoch={epoch}/{epochs}: " + "; ".join(f"{name} order={rows[name][-1]['order']:.6f} weighted_l2={rows[name][-1]['weighted_l2']:.6f}" for name in names), flush=True)
    after = representation_state(model)
    assert all(torch.equal(before[key], after[key]) for key in before)
    assert all(not getattr(model, name).training and all(not p.requires_grad and p.grad is None for p in getattr(model, name).parameters())
               for name in ("sensor_encoder", "conditional_prior", "decoder"))
    assert hashlib.sha256(Path(representation_path).read_bytes()).hexdigest() == source_hash

    # Ground truth is first read after ALL heads finish; never used for training.
    references = {}
    for split, (_, split_units, split_cycles) in data.items():
        table = read_reference(config["data"][f"hi_{split}_path"], processed[split]).set_index(["unit", "cycle"])
        keys = list(zip(split_units.cpu().tolist(), split_cycles.cpu().tolist()))
        references[split] = table.loc[keys, "hi_gt"].to_numpy()
    logs, predictions, comparison = {}, {}, []
    for name in names:
        folder = run_dir / name
        folder.mkdir()
        logs[name] = pd.DataFrame(rows[name])
        logs[name].to_csv(folder / "losses.csv", index=False, float_format="%.17g")
        head = heads[name].eval()
        state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        assert any(not torch.equal(state[key], initial_states[name][key]) for key in state)
        saved = {"state_dict": state, "initial_state_dict": initial_states[name], "transform": transforms[name],
                 "experiment": name, "mode": mode, "epoch": epochs, "source_sha256": source_hash,
                 "input_dim": features[name]["dev"][0].shape[1], "head_parameters": parameter_counts[name],
                 "source_checkpoint": str(representation_path), "seed": seed,
                 "config": config, "normalization": source["normalization"], "columns": source["columns"],
                 "pair_batches_sha256": digests[name].hexdigest()}
        torch.save(saved, folder / "head.pt")
        checkpoint = torch.load(folder / "head.pt", map_location="cpu", weights_only=True)
        restored = HIHead(checkpoint["input_dim"], 16).to(device).eval()
        restored.load_state_dict(checkpoint["state_dict"], strict=True)
        tables = []
        for split, (values, split_units, split_cycles) in features[name].items():
            with torch.no_grad():
                scores = head(inputs[name][split])
                reloaded = restored(apply_transform(values, checkpoint["transform"]))
            torch.testing.assert_close(scores, reloaded, rtol=0, atol=0)
            assert torch.isfinite(scores).all()
            tables.append(pd.DataFrame({"experiment": name, "split": split, "unit": split_units.cpu().numpy(), "cycle": split_cycles.cpu().numpy(),
                                        "r": scores.cpu().numpy(), "hi_raw": torch.sigmoid(-scores).cpu().numpy(), "hi_gt": references[split]}))
            comparison += [{**row, "head_parameters": parameter_counts[name]} for row in
                           ranking_rows(name, split, scores, split_units, split_cycles, references[split], settings["min_cycle_gap"], settings["kappa"])]
        predictions[name] = pd.concat(tables, ignore_index=True)
        predictions[name].to_csv(folder / "predictions.csv", index=False, float_format="%.17g")
    pd.DataFrame(comparison).to_csv(run_dir / "comparison.csv", index=False, float_format="%.17g")
    plot_results(run_dir, logs, predictions, mode, diagnostic)
    verification = {"diagnostic": diagnostic, "mode": mode, "epochs": epochs, "device": str(device), "source_sha256": source_hash,
                    "source_checkpoint": str(representation_path), "seed": seed, "head_parameters": parameter_counts,
                    "representation_frozen_exact": True, "identical_initial_8d_heads": True,
                    "identical_initial_head_groups": [list(names[:2])],
                    "identical_pair_batches": True, "sampling_rng_isolated": True,
                    "pair_checks_by_epoch": epoch_checks,
                    "dev_only_ddof_zero_fit": True, "exact_reload_predictions_both_splits": True,
                    "all_valid_pairs_enumerated": True, "labels_read_after_training": True,
                    "feature_keys_verified": True,
                    "blocks_concat_standardization_exact": experiment == "blocks",
                    "zero_std_dimensions": zero_dimensions,
                    "pair_batches_sha256": digests[names[0]].hexdigest()}
    (run_dir / "verification.json").write_text(json.dumps(verification, indent=2, allow_nan=False), encoding="utf-8")
    print(pd.DataFrame(comparison).query("unit == 'all'").to_string(index=False), flush=True)
    print(f"PASS: {diagnostic} {mode} fairness, freezing, dev-only transform, exact reload and exhaustive ranking; output={run_dir}", flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--experiment", required=True, choices=["scale", "blocks"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(load_config(args.config), args.representation, args.smoke, args.experiment)


if __name__ == "__main__":
    main()
