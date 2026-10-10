"""Keyed HI/RUL evaluation with train-only scaling and raw/scaled curves."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import FlightDataset, load_config, load_processed, read_reference, read_rul
from train import extract_training_means, load_ocald, scale_hi


def attach_references(predictions, hi, rul):
    keys = ["unit", "cycle"]
    for name, table in (("predictions", predictions), ("HI", hi), ("RUL", rul)):
        if table.duplicated(keys).any():
            raise ValueError(f"Duplicate {name} unit/cycle keys")
    expected = set(map(tuple, predictions[keys].to_numpy()))
    if set(map(tuple, hi[keys].to_numpy())) != expected or set(map(tuple, rul[keys].to_numpy())) != expected:
        raise ValueError("Missing or unexpected HI/RUL associations")
    result = predictions.merge(hi, on=keys, validate="one_to_one").merge(rul, on=keys, validate="one_to_one")
    if result[["hi_raw", "hi_gt", "rul"]].isna().any().any():
        raise ValueError("Missing prediction or evaluation reference values")
    return result.sort_values(keys).reset_index(drop=True)


def predict_table(config, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(1)
    model, checkpoint = load_ocald(checkpoint_path, device)
    if checkpoint["config"] != config:
        raise ValueError("Evaluation configuration differs from checkpoint")
    processed = load_processed(config)
    if checkpoint["columns"] != processed["columns"]:
        raise ValueError("Checkpoint columns differ from prepared data")
    for field in ("x", "w"):
        for key in ("min", "max"):
            if not torch.equal(checkpoint["normalization"][field][key], processed["normalization"][field][key]):
                raise ValueError("Checkpoint normalization differs from prepared training normalization")
    parameters = checkpoint["hi_minmax"]
    scaled_available = parameters["max"] != parameters["min"]
    tables = []
    for split in ("dev", "test"):
        dataset = FlightDataset(processed, split)
        means, units, cycles = extract_training_means(model, dataset, config["training"]["representation"]["batch_size"], device)
        with torch.no_grad():
            raw = torch.sigmoid(-model.hi_head(means)).cpu()
        if not torch.isfinite(raw).all() or not ((raw > 0) & (raw < 1)).all():
            raise ValueError(f"Invalid raw HI in {split}")
        if split == "dev":
            torch.testing.assert_close(raw, checkpoint["verification"]["train_hi_raw"], rtol=1e-6, atol=1e-7)
            stored = checkpoint["verification"]["train_hi_raw"]
            if parameters != {"min": stored.min().item(), "max": stored.max().item(), "count": len(dataset), "fit_split": "dev"}:
                raise ValueError("HI scale is not fitted on all training predictions")
        scaled = scale_hi(raw, parameters).numpy() if scaled_available else np.full(len(raw), np.nan)
        predictions = pd.DataFrame({"split": split, "unit": units.cpu().numpy(), "cycle": cycles.cpu().numpy(),
                                    "hi_raw": raw.numpy(), "hi_scaled": scaled})
        hi = read_reference(config["data"][f"hi_{split}_path"], processed[split])
        rul = read_rul(config["data"]["raw_path"], split, processed[split])
        tables.append(attach_references(predictions, hi, rul))
    return pd.concat(tables, ignore_index=True), checkpoint


def metric_report(table, checkpoint):
    from metrics import evaluate_metrics

    result = {"stage": "evaluation", "mode": checkpoint["mode"], "source_mode": checkpoint["source_mode"],
              "hi_epoch": checkpoint["epoch"], "representation_epoch": checkpoint["source_epoch"],
              "seed": checkpoint["config"]["training"]["seed"], "hi_minmax": checkpoint["hi_minmax"],
              "reference_commit": "157ebfae7baeee1e0cd52180f8e58ead3dcb448e", "splits": {}}
    for split in ("dev", "test"):
        subset = table.loc[table.split == split]
        scales = {}
        for name, column in (("raw", "hi_raw"), ("scaled", "hi_scaled")):
            if subset[column].isna().all():
                scales[name] = {"available": False, "reason": "Training HI range is zero"}
            else:
                if subset[column].isna().any():
                    raise ValueError("Partially missing HI scale")
                scales[name] = {"available": True, **evaluate_metrics(subset[column].to_numpy(), subset.hi_gt.to_numpy(),
                                                                        subset.unit.to_numpy(), subset.cycle.to_numpy(),
                                                                        subset.rul.to_numpy(), result["seed"])}
        result["splits"][split] = scales
    return result


def plot_curves(table, report, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def label(value):
        return "N/A" if value is None else f"{value:.3f}"

    groups = list(table.groupby(["split", "unit"], sort=True))
    figure, axes = plt.subplots(len(groups), 2, figsize=(14, 2.5 * len(groups)), layout="constrained", squeeze=False)
    figure.suptitle(f"DS03 HI — {report['mode']} | Stage 1 {report['representation_epoch']} / Stage 2 {report['hi_epoch']} epochs")
    try:
        for row, ((split, unit), group) in enumerate(groups):
            group = group.sort_values("cycle")
            for column, (scale, field) in enumerate((("raw", "hi_raw"), ("scaled", "hi_scaled"))):
                axis = axes[row, column]
                axis.plot(group.cycle, group.hi_gt, color="black", linewidth=1, label="Reference HI")
                entry = report["splits"][split][scale]
                if entry["available"]:
                    line, = axis.plot(group.cycle, group[field], linewidth=1.1, label=f"Predicted HI ({scale})")
                    if not np.array_equal(line.get_xdata(), group.cycle.to_numpy()) or not np.array_equal(line.get_ydata(), group[field].to_numpy()):
                        raise ValueError("Curve coordinates differ from prediction table")
                    metrics = entry["per_unit"][str(int(unit))]
                    details = f"MAE={label(metrics['mae']['value'])} MAPE%={label(metrics['mape']['value'])} M={label(metrics['monotonicity']['value'])} rho={label(metrics['spearman']['value'])} MI={label(metrics['mi']['value'])}"
                else:
                    details = "Scaled unavailable: zero training HI range"
                axis.set_title(f"{split} unit {int(unit)} | {scale}\n{details}", fontsize=9)
                axis.set_xlabel("Flight cycle")
                axis.set_ylabel("HI")
                axis.grid(alpha=0.25)
                axis.legend(fontsize=8)
        figure.savefig(output_path, dpi=140)
    finally:
        plt.close(figure)


def evaluate(config, checkpoint_path):
    table, checkpoint = predict_table(config, checkpoint_path)
    run_dir = Path(checkpoint_path).parent
    csv_path = run_dir / "predictions.csv"
    table.to_csv(csv_path, index=False, float_format="%.17g")
    # Render from the exported table to preserve exact CSV/curve agreement.
    exported = pd.read_csv(csv_path, float_precision="round_trip")
    report = metric_report(exported, checkpoint)
    plot_curves(exported, report, run_dir / "hi_curves.png")
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Evaluation mode={checkpoint['mode']}: {len(table)} rows; output={run_dir}")
    for split in ("dev", "test"):
        for scale, entry in report["splits"][split].items():
            print(split, scale, entry.get("overall", entry))
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    evaluate(load_config(args.config), args.checkpoint)


if __name__ == "__main__":
    main()
