"""Task-scoped verification. Failures always return a nonzero exit code."""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from data import (FlightDataset, collate_fn, inspect_raw, load_config,
                  load_processed, print_summary, read_reference)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify_data(config):
    processed = load_processed(config)
    settings = config["data"]
    stride = settings["downsample_stride"]
    extrema = {}
    with h5py.File(settings["raw_path"], "r") as hdf:
        columns, layouts = inspect_raw(hdf, settings)
        check(processed["columns"] == columns, "Saved column names differ from raw file")
        for split in ("dev", "test"):
            flights = processed[split]
            keys = [(f["unit"], f["cycle"]) for f in flights]
            expected_keys = [(block["unit"], cycle) for block in layouts[split]
                             for cycle, _, _ in block["flights"]]
            check(keys == expected_keys and len(keys) == len(set(keys)), f"Flight keys/order mismatch: {split}")
            by_key = dict(zip(keys, flights))
            # Verify every selected raw X/W element, with independently built
            # unit-level row indices; no reconstruction via prepare_data().
            for block in layouts[split]:
                start, stop = block["start"], block["stop"]
                unit_rows = np.arange(start, stop, stride)
                raw = {"x": hdf[f"X_s_{split}"][start:stop],
                       "w": hdf[f"W_{split}"][start:stop]}
                for field, values in raw.items():
                    check(np.isfinite(values).all(), f"Nonfinite raw values: {split}/{field}")
                    if split == "dev":
                        lo, hi = values.min(axis=0), values.max(axis=0)
                        if field in extrema:
                            lo = np.minimum(lo, extrema[field][0])
                            hi = np.maximum(hi, extrema[field][1])
                        extrema[field] = (lo, hi)
                for cycle, left, right in block["flights"]:
                    flight = by_key[(block["unit"], cycle)]
                    rows = unit_rows[(unit_rows >= left) & (unit_rows < right)]
                    check(len(rows) > 0, "Empty sampled flight")
                    check((flight["raw_start"], flight["raw_stop"], flight["first_sample_row"])
                          == (left, right, int(rows[0])), "Raw boundary/sampling phase mismatch")
                    for field, dimension in (("x", settings["sensor_dim"]), ("w", settings["condition_dim"])):
                        actual = flight[field]
                        check(actual.dtype == torch.float32 and actual.shape == (len(rows), dimension),
                              f"Invalid dtype/length/dimension: {split}/{block['unit']}/{cycle}/{field}")
                        params = processed["normalization"][field]
                        lo, hi = params["min"].numpy(), params["max"].numpy()
                        denominator = hi - lo
                        denominator[denominator == 0] = 1.0
                        expected = ((raw[field][rows - start] - lo) / denominator).astype(np.float32)
                        check(np.isfinite(actual.numpy()).all() and np.array_equal(actual.numpy(), expected),
                              f"Raw alignment or normalization mismatch: {split}/{block['unit']}/{cycle}/{field}")
        for field, (lo, hi) in extrema.items():
            params = processed["normalization"][field]
            check(np.array_equal(params["min"].numpy(), lo) and np.array_equal(params["max"].numpy(), hi),
                  f"Normalization was not fitted on all raw dev rows: {field}")
            constants = np.flatnonzero(lo == hi).tolist()
            print(f"{field}: train-only raw-row Min–Max verified; constant columns={constants}")

    check(set(processed) == {"dev", "test", "normalization", "columns", "data_settings"},
          "Unexpected processed data fields")
    train_units = {f["unit"] for f in processed["dev"]}
    test_units = {f["unit"] for f in processed["test"]}
    check(not train_units & test_units, "dev/test overlap")
    for split in ("dev", "test"):
        dataset = FlightDataset(processed, split)
        # All flights pass through collate in bounded batches, including longest.
        for start in range(0, len(dataset), 16):
            samples = [dataset[i] for i in range(start, min(start + 16, len(dataset)))]
            batch = collate_fn(samples)
            check(set(batch) == {"x", "w", "mask", "unit", "cycle"}, "Training batch contains extra fields")
            check(batch["mask"].dtype == torch.bool and batch["unit"].dtype == batch["cycle"].dtype == torch.int64,
                  "Incorrect mask or identifier dtype")
            for j, sample in enumerate(samples):
                length = len(sample["x"])
                check(int(batch["mask"][j].sum()) == length and batch["mask"][j, :length].all()
                      and not batch["mask"][j, length:].any(), "Mask length/position mismatch")
                check(batch["unit"][j].item() == sample["unit"] and batch["cycle"][j].item() == sample["cycle"],
                      "Batch identifier mismatch")
                for field in ("x", "w"):
                    check(torch.equal(batch[field][j, :length], sample[field])
                          and not torch.count_nonzero(batch[field][j, length:]), "Truncation or incorrect padding")
        # Explicitly test that true zeros remain valid; mask is length-derived.
        zeros = {"x": torch.zeros(2, 14), "w": torch.zeros(2, 4), "unit": 1, "cycle": 1}
        zero_batch = collate_fn([zeros, dataset[0]])
        check(zero_batch["mask"][0, :2].all() and int(zero_batch["mask"][0].sum()) == 2,
              "Zero-valued valid rows were mistaken for padding")
        refs = read_reference(settings[f"hi_{split}_path"], processed[split])
        expected = {(f["unit"], f["cycle"]) for f in processed[split]}
        check(set(zip(refs.unit, refs.cycle)) == expected, "Reference key alignment mismatch")
        print(f"{split}: HI alignment verified for {len(refs)} flights (per-unit stored row order, cycles 1..N)")
        print(f"{split}: mask, X/W alignment, no truncation and label-free batches verified")
        below = sum(int((f["x"] < 0).sum() + (f["w"] < 0).sum()) for f in processed[split])
        above = sum(int((f["x"] > 1).sum() + (f["w"] > 1).sum()) for f in processed[split])
        print(f"{split}: elements below 0={below}, above 1={above}; no clipping")
    print_summary(processed)
    print("PASS: T1 real DS03 data verification (not smoke or training)")


def plot_data(config, split, unit, cycle, parameters):
    """Plot one sampled flight and the unit's full reference HI trajectory."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    processed = load_processed(config)
    flights = processed[split]
    selected = [f for f in flights if f["unit"] == unit and f["cycle"] == cycle]
    check(len(selected) == 1, f"Expected one flight for split={split}, unit={unit}, cycle={cycle}")
    check(len(parameters) == len(set(parameters)), "Duplicate parameter names")
    columns = processed["columns"]
    lookup = {}
    for field in ("x", "w"):
        for index, name in enumerate(columns[field]):
            check(name not in lookup, f"Ambiguous X/W parameter name: {name}")
            lookup[name] = (field, index)
    unknown = set(parameters) - set(lookup)
    check(not unknown, f"Unknown parameters: {sorted(unknown)}; available: {list(lookup)}")
    reference = read_reference(config["data"][f"hi_{split}_path"], flights)
    reference = reference.loc[reference.unit == unit].sort_values("cycle")
    expected_cycles = sorted(f["cycle"] for f in flights if f["unit"] == unit)
    check(reference.cycle.tolist() == expected_cycles, "Incomplete unit HI trajectory")
    flight = selected[0]
    length = len(flight["x"])
    check(length == len(flight["w"]), "X/W flight length mismatch")
    positions = (flight["first_sample_row"] - flight["raw_start"]
                 + np.arange(length) * config["data"]["downsample_stride"])
    figure, axes = plt.subplots(len(parameters) + 1, 1,
                                figsize=(11, 2.5 * (len(parameters) + 1)), layout="constrained")
    figure.suptitle(f"DS03 {split} | unit {unit} | flight cycle {cycle}")
    try:
        for axis, name in zip(axes[:-1], parameters):
            field, index = lookup[name]
            values = flight[field][:, index].numpy()
            check(np.isfinite(values).all(), f"Nonfinite parameter: {name}")
            line, = axis.plot(positions, values, linewidth=1.1)
            check(np.array_equal(line.get_xdata(), positions) and np.array_equal(line.get_ydata(), values),
                  f"Plot data mismatch: {name}")
            axis.set_title(f"{name} ({'sensor X_s' if field == 'x' else 'condition W'}) — {length} sampled points")
            axis.set_xlabel("Original row position within flight (sampling stride 10)")
            axis.set_ylabel("Train Min–Max value")
            axis.grid(alpha=0.25)
        axis = axes[-1]
        line, = axis.plot(reference.cycle.to_numpy(), reference.hi_gt.to_numpy(),
                          marker=".", markersize=4, linewidth=1.2, label="Reference HI (ground truth)")
        check(np.array_equal(line.get_xdata(), reference.cycle.to_numpy())
              and np.array_equal(line.get_ydata(), reference.hi_gt.to_numpy()), "HI plot data mismatch")
        axis.axvline(cycle, color="tab:orange", linestyle="--", label=f"Selected flight: cycle {cycle}")
        axis.set_title(f"Full unit {unit} reference HI — {len(reference)} flights")
        axis.set_xlabel("Flight cycle")
        axis.set_ylabel("Reference HI")
        axis.grid(alpha=0.25)
        axis.legend()
        output = Path(config["output"]["root"]) / "data_inspection" / f"{split}_unit_{unit}_cycle_{cycle}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160)
    finally:
        plt.close(figure)
    print(f"Flight: split={split}, unit={unit}, cycle={cycle}, points={length}, parameters={parameters}")
    print(f"Reference HI: {len(reference)} flights, cycles {expected_cycles[0]}..{expected_cycles[-1]}")
    print(f"Plot saved: {output}")
    print("PASS: T1 data plot (normalized flight parameters and full reference HI)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["data", "data-plot"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["dev", "test"])
    parser.add_argument("--unit", type=int)
    parser.add_argument("--cycle", type=int)
    parser.add_argument("--parameters", nargs="+", help="X_s/W variable names to plot")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.task == "data":
        verify_data(config)
    else:
        if any(value is None for value in (args.split, args.unit, args.cycle, args.parameters)):
            parser.error("data-plot requires --split, --unit, --cycle and --parameters")
        plot_data(config, args.split, args.unit, args.cycle, args.parameters)


if __name__ == "__main__":
    main()
