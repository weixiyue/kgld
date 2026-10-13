"""Keyed HI/RUL evaluation with train-only scaling and raw/scaled curves."""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import FlightDataset, load_config, load_processed, read_reference, read_rul
from train import extract_training_means, load_ocald, scale_hi, fit_score_scale, scale_score


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


def predict_table(config=None, checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(1)
    model, checkpoint = load_ocald(checkpoint_path, device)
    # The saved snapshot is authoritative. The first argument remains for old callers.
    config = checkpoint['config']
    diagnostic = checkpoint.get('stage') in ('joint', 'ocald_no_kli', 'ocald_s2')
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
    scalar = model.score_mode == "scalar_i"
    score_parameters = None
    for split in ("dev", "test"):
        dataset = FlightDataset(processed, split)
        if diagnostic and "flight_keys" in checkpoint:
            lookup = {(f["unit"], f["cycle"]): f for f in dataset.flights}
            keys = [tuple(k) for k in checkpoint["flight_keys"][split]]
            if len(set(keys)) != len(keys):
                raise ValueError("Duplicate diagnostic flight keys")
            dataset.flights = [lookup[key] for key in keys]
        means, units, cycles = extract_training_means(model, dataset, config["training"]["representation"]["batch_size"], device)
        with torch.no_grad():
            device_scores = model.score(means)
            scores = device_scores.cpu()
            raw = scores if scalar else torch.sigmoid(-device_scores).cpu()
        in_range = ((raw >= 0) & (raw <= 1)) if diagnostic else ((raw > 0) & (raw < 1))
        if not torch.isfinite(raw).all() or (not scalar and not in_range.all()):
            raise ValueError(f"Invalid raw HI in {split}")
        if split == "dev":
            stored = checkpoint["raw_hi"]["dev"] if diagnostic else checkpoint["verification"]["train_hi_raw"]
            torch.testing.assert_close(raw, stored, rtol=1e-6, atol=1e-7)
            if parameters != {"min": stored.min().item(), "max": stored.max().item(), "count": len(dataset), "fit_split": "dev"}:
                raise ValueError("HI scale is not fitted on all training predictions")
        if split == 'dev':
            score_parameters = fit_score_scale(scores)
            if 'score_minmax' in checkpoint and checkpoint['score_minmax'] != score_parameters:
                raise ValueError('Checkpoint score scale differs from dev-only predictions')
            checkpoint = {**checkpoint, 'score_minmax': score_parameters, 'score_mode': model.score_mode}
        score_scaled = scale_score(scores, score_parameters).numpy() if score_parameters['available'] else np.full(len(scores), np.nan)
        scaled = np.full(len(raw), np.nan) if scalar else (scale_hi(raw, parameters).numpy() if scaled_available else np.full(len(raw), np.nan))
        predictions = pd.DataFrame({"split": split, "unit": units.cpu().numpy(), "cycle": cycles.cpu().numpy(),
                                    "hi_raw": raw.numpy(), "hi_scaled": scaled, "hi_score": scores.numpy(),
                                    "hi_score_scaled": score_scaled})
        hi = read_reference(config["data"][f"hi_{split}_path"], processed[split])
        rul = read_rul(config["data"]["raw_path"], split, processed[split])
        if diagnostic and "flight_keys" in checkpoint:
            keys = pd.MultiIndex.from_arrays([predictions.unit, predictions.cycle], names=["unit", "cycle"])
            hi = hi.set_index(["unit", "cycle"]).loc[keys].reset_index()
            rul = rul.set_index(["unit", "cycle"]).loc[keys].reset_index()
        tables.append(attach_references(predictions, hi, rul))
    return pd.concat(tables, ignore_index=True), checkpoint


def metric_report(table, checkpoint):
    from metrics import evaluate_metrics

    diagnostic = checkpoint.get("stage") in ("joint", "ocald_no_kli", "ocald_s2")
    if diagnostic:
        result = {"stage": "diagnostic_evaluation", "diagnostic_type": checkpoint["stage"],
                  "mode": checkpoint["mode"], "joint_epoch": checkpoint["epoch"],
                  "seed": checkpoint["seed"], "hi_minmax": checkpoint["hi_minmax"],
                  "actual_training": checkpoint['actual_training'] if 'actual_training' in checkpoint else
                                      {"lambda": checkpoint["lambda"], "kl_i_weight": 1., "kl_c_weight": 1., "seed": checkpoint["seed"]},
                  "reference_commit": "157ebfae7baeee1e0cd52180f8e58ead3dcb448e", "splits": {}}
    else:
        result = {"stage": "evaluation", "mode": checkpoint["mode"], "source_mode": checkpoint["source_mode"],
              "hi_epoch": checkpoint["epoch"], "representation_epoch": checkpoint["source_epoch"],
              "seed": checkpoint["config"]["training"]["seed"], "hi_minmax": checkpoint["hi_minmax"],
              "reference_commit": "157ebfae7baeee1e0cd52180f8e58ead3dcb448e", "splits": {}}
    result.update(score_mode=checkpoint.get("score_mode", "hi_head"), score_minmax=checkpoint.get("score_minmax"),
                  hi_raw_kind="score" if checkpoint.get("score_mode") == "scalar_i" else "sigmoid_hi",
                  final_hi="hi_score_scaled" if checkpoint.get("score_mode") == "scalar_i" else "hi_scaled")
    for split in ("dev", "test"):
        subset = table.loc[table.split == split]
        scales = {}
        for name, column in (("raw", "hi_raw"), ("scaled", "hi_scaled"), ("score_scaled", "hi_score_scaled")):
            if column not in subset:
                continue  # Historical CSVs retain their original two scales.
            if checkpoint.get("score_mode") == "scalar_i" and name == "raw":
                scales[name] = {"available": False, "reason": "Raw scalar I is degradation score r; final HI uses score_scaled"}
                continue
            if subset[column].isna().all():
                scales[name] = {"available": False, "reason": "Scalar I bypasses Sigmoid; use score_scaled" if checkpoint.get("score_mode") == "scalar_i" and name == "scaled" else "Dev prediction range is zero"}
            else:
                if subset[column].isna().any():
                    raise ValueError("Partially missing HI scale")
                if diagnostic:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        values = evaluate_metrics(subset[column].to_numpy(), subset.hi_gt.to_numpy(),
                                                  subset.unit.to_numpy(), subset.cycle.to_numpy(),
                                                  subset.rul.to_numpy(), result["seed"])
                    for entries in [values["overall"], *values["per_unit"].values()]:
                        for entry in entries.values():
                            if not np.isfinite(entry["value"]):
                                original = entry.get("reason")
                                entry["value"] = None
                                entry["reason"] = "Original metrics formula returned a nonfinite value; no epsilon or clipping" + (f"; {original}" if original else "")
                    scales[name] = {"available": True, **values, "warnings": sorted({str(w.message) for w in caught})}
                else:
                    scales[name] = {"available": True, **evaluate_metrics(subset[column].to_numpy(), subset.hi_gt.to_numpy(),
                                                                        subset.unit.to_numpy(), subset.cycle.to_numpy(),
                                                                        subset.rul.to_numpy(), result["seed"])}
        result["splits"][split] = scales
        if diagnostic:
            result["splits"][split]["saturation"] = {"zero": int((subset.hi_raw == 0).sum()),
                                                     "one": int((subset.hi_raw == 1).sum()), "count": len(subset)}
    return result


def plot_curves(table, report, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def label(value):
        return "N/A" if value is None else f"{value:.3f}"

    groups = list(table.groupby(["split", "unit"], sort=True))
    curve_scales = [("raw", "hi_raw"), ("scaled", "hi_scaled")]
    if "hi_score_scaled" in table and "score_scaled" in report["splits"]["dev"]:
        curve_scales.append(("score_scaled", "hi_score_scaled"))
    figure, axes = plt.subplots(len(groups), len(curve_scales), figsize=(7 * len(curve_scales), 2.5 * len(groups)), layout="constrained", squeeze=False)
    if "diagnostic_type" in report:
        figure.suptitle(f"DS03 HI — {report['diagnostic_type']} | {report['mode']} | {report['joint_epoch']} joint epochs")
    else:
        figure.suptitle(f"DS03 HI — {report['mode']} | Stage 1 {report['representation_epoch']} / Stage 2 {report['hi_epoch']} epochs")
    try:
        for row, ((split, unit), group) in enumerate(groups):
            group = group.sort_values("cycle")
            for column, (scale, field) in enumerate(curve_scales):
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
                    details = entry["reason"]
                axis.set_title(f"{split} unit {int(unit)} | {scale}\n{details}", fontsize=9)
                axis.set_xlabel("Flight cycle")
                axis.set_ylabel("HI")
                axis.grid(alpha=0.25)
                axis.legend(fontsize=8)
        figure.savefig(output_path, dpi=140)
    finally:
        plt.close(figure)


def evaluate(config, checkpoint_path, output_dir=None):
    table, checkpoint = predict_table(config, checkpoint_path)
    run_dir = Path(checkpoint_path).parent if output_dir is None else Path(output_dir)
    if output_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=False)
    if checkpoint.get("stage") in ("joint", "ocald_no_kli", "ocald_s2"):
        if any((run_dir / name).exists() for name in ("predictions.csv", "metrics.json", "hi_curves.png")):
            raise ValueError("Refusing to overwrite diagnostic evaluation artifacts; supply a new output directory")
    csv_path = run_dir / "predictions.csv"
    # Separate evaluation metadata preserves old checkpoint and artifacts.
    (run_dir / "score_scaling.json").write_text(json.dumps({"score_mode": checkpoint["score_mode"],
        "score_minmax": checkpoint["score_minmax"], "source_checkpoint": str(checkpoint_path)}, indent=2) + "\n")
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
    parser.add_argument("--config", help="Legacy argument; evaluation uses the checkpoint configuration snapshot")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", help="Export to a new directory, e.g. for read-only diagnostic comparison")
    args = parser.parse_args()
    evaluate(None, args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
