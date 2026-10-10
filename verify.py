"""Task-scoped verification. Failures always return a nonzero exit code."""

import argparse

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["data"])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    verify_data(load_config(args.config))


if __name__ == "__main__":
    main()
