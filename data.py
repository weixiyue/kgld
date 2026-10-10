"""DS03 full-flight data; evaluation references have a separate read path."""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    d = config["data"]
    required = {
        "sensor_field": "X_s", "condition_field": "W",
        "sensor_dim": 14, "condition_dim": 4,
        "downsample_stride": 10, "downsample_scope": "unit",
        "sample_scope": "full_flight", "normalization": "minmax_0_1",
        "normalization_fit": "raw_dev_before_downsampling",
        "normalization_clip": False,
    }
    for key, value in required.items():
        if key not in d or d[key] != value:
            raise ValueError(f"Unapproved or missing data setting: {key} (expected {value!r})")
    return config


def variable_names(dataset):
    return [v.decode("utf-8") for v in dataset[:]]


def integer_ids(values, name):
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{name} must contain finite integers")
    result = values.astype(np.int64)
    if (result < 1).any():
        raise ValueError(f"{name} must be positive")
    return result


def inspect_raw(hdf, settings):
    """Read only unit/cycle auxiliary columns; never load HI, RUL or T."""
    names = {"x": variable_names(hdf["X_s_var"]),
             "w": variable_names(hdf["W_var"]),
             "auxiliary": variable_names(hdf["A_var"])}
    for key in names:
        if len(names[key]) != len(set(names[key])):
            raise ValueError(f"Duplicate variable names: {key}")
    if len(names["x"]) != settings["sensor_dim"] or len(names["w"]) != settings["condition_dim"]:
        raise ValueError("Actual X_s/W variable dimensions differ from approved dimensions")
    aux_names = names["auxiliary"]
    layouts = {}
    for split in ("dev", "test"):
        a, x, w = (hdf[f"{field}_{split}"] for field in ("A", "X_s", "W"))
        if a.ndim != 2 or a.shape[1] != len(aux_names):
            raise ValueError(f"Invalid auxiliary shape: {split}")
        if x.shape != (a.shape[0], settings["sensor_dim"]) or w.shape != (a.shape[0], settings["condition_dim"]):
            raise ValueError(f"X/W/A row or column mismatch: {split}")
        unit = integer_ids(a[:, aux_names.index("unit")], "unit")
        cycle = integer_ids(a[:, aux_names.index("cycle")], "cycle")
        cuts = np.r_[0, np.flatnonzero(unit[1:] != unit[:-1]) + 1, len(unit)]
        blocks = []
        seen = set()
        for start, stop in zip(cuts[:-1], cuts[1:]):
            u = int(unit[start])
            if u in seen:
                raise ValueError(f"Noncontiguous/repeated unit block: {split}, unit={u}")
            seen.add(u)
            c = cycle[start:stop]
            if (np.diff(c) < 0).any():
                raise ValueError(f"Cycle reset or unordered rows: {split}, unit={u}")
            boundaries = np.r_[0, np.flatnonzero(c[1:] != c[:-1]) + 1, len(c)]
            flights = [(int(c[left]), int(start + left), int(start + right))
                       for left, right in zip(boundaries[:-1], boundaries[1:])]
            blocks.append({"unit": u, "start": int(start), "stop": int(stop), "flights": flights})
        if not blocks:
            raise ValueError(f"Empty split: {split}")
        layouts[split] = blocks
    overlap = {b["unit"] for b in layouts["dev"]} & {b["unit"] for b in layouts["test"]}
    if overlap:
        raise ValueError(f"dev/test unit overlap: {sorted(overlap)}")
    return names, layouts


def normalize(values, parameters):
    minimum = parameters["min"].numpy()
    span = parameters["max"].numpy() - minimum
    # MinMaxScaler's constant-column convention: denominator 1, no clipping.
    denominator = np.where(span == 0, 1.0, span)
    return ((values - minimum) / denominator).astype(np.float32)


def prepare_data(config):
    """Fit on ALL raw dev rows, then sample continuously within each unit."""
    settings = config["data"]
    result = {"dev": [], "test": [], "normalization": {}, "data_settings": dict(settings)}
    with h5py.File(settings["raw_path"], "r") as hdf:
        names, layouts = inspect_raw(hdf, settings)
        result["columns"] = names
        # One unit at a time bounds memory; unrelated HDF5 arrays are not read.
        for split in ("dev", "test"):
            for block in layouts[split]:
                start, stop = block["start"], block["stop"]
                raw = {"x": hdf[f"X_s_{split}"][start:stop],
                       "w": hdf[f"W_{split}"][start:stop]}
                for key, values in raw.items():
                    if not np.isfinite(values).all():
                        raise ValueError(f"Nonfinite {key}: {split}, unit={block['unit']}")
                    if split == "dev":
                        lo, hi = values.min(axis=0), values.max(axis=0)
                        params = result["normalization"].get(key)
                        if params is not None:
                            lo = np.minimum(lo, params["min"].numpy())
                            hi = np.maximum(hi, params["max"].numpy())
                        result["normalization"][key] = {"min": torch.from_numpy(lo), "max": torch.from_numpy(hi)}
                stride = settings["downsample_stride"]
                sampled = {key: values[::stride].copy() for key, values in raw.items()}
                del raw
                for cycle, left, right in block["flights"]:
                    # These indices keep the unit's sampling phase across flights.
                    sample_left = (left - start + stride - 1) // stride
                    sample_right = (right - start + stride - 1) // stride
                    if sample_right <= sample_left:
                        raise ValueError(f"Downsampling removed flight: {split}, unit={block['unit']}, cycle={cycle}")
                    result[split].append({
                        "unit": block["unit"], "cycle": cycle,
                        "raw_start": left, "raw_stop": right,
                        "first_sample_row": start + sample_left * stride,
                        "x": torch.from_numpy(sampled["x"][sample_left:sample_right].copy()),
                        "w": torch.from_numpy(sampled["w"][sample_left:sample_right].copy()),
                    })
    # Conversion occurs after the final dev extrema have been obtained.
    for split in ("dev", "test"):
        for flight in result[split]:
            for key in ("x", "w"):
                flight[key] = torch.from_numpy(normalize(flight[key].numpy(), result["normalization"][key]))
    return result


def load_processed(config):
    data = torch.load(config["data"]["processed_path"], map_location="cpu", weights_only=True)
    if data["data_settings"] != config["data"]:
        raise ValueError("Processed data settings differ from config; rerun --prepare")
    return data


class FlightDataset(Dataset):
    def __init__(self, processed, split):
        if split not in ("dev", "test"):
            raise ValueError("split must be dev or test")
        self.flights = processed[split]

    def __len__(self):
        return len(self.flights)

    def __getitem__(self, index):
        flight = self.flights[index]
        return {key: flight[key] for key in ("x", "w", "unit", "cycle")}


def collate_fn(flights):
    if not flights:
        raise ValueError("Cannot collate an empty batch")
    lengths = torch.tensor([len(f["x"]) for f in flights], dtype=torch.int64)
    if (lengths < 1).any() or any(len(f["w"]) != len(f["x"]) for f in flights):
        raise ValueError("Invalid or mismatched X/W lengths")
    batch = {
        "x": torch.nn.utils.rnn.pad_sequence([f["x"] for f in flights], batch_first=True),
        "w": torch.nn.utils.rnn.pad_sequence([f["w"] for f in flights], batch_first=True),
        "mask": torch.arange(int(lengths.max()))[None, :] < lengths[:, None],
        "unit": torch.tensor([f["unit"] for f in flights], dtype=torch.int64),
        "cycle": torch.tensor([f["cycle"] for f in flights], dtype=torch.int64),
    }
    return batch


def read_reference(path, flights):
    """Read the observed pandas fixed HDF layout with h5py, without PyTables.

    Absent cycle fields use the reference notebook's per-unit row convention,
    but only after checking the stored row index, device blocks and cycle counts.
    """
    with h5py.File(path, "r") as hdf:
        group = hdf["df"]
        if group.attrs["pandas_type"] != b"frame":
            raise ValueError("Unsupported reference format; expected pandas fixed frame")
        columns = variable_names(group["axis0"])
        index = group["axis1"][:]
        if index.ndim != 1 or len(index) == 0 or not np.equal(index, np.floor(index)).all() or not (np.diff(index) == 1).all():
            raise ValueError("Reference row index is not consecutive in stored order")
        values = {}
        for block in range(int(group.attrs["nblocks"])):
            block_names = variable_names(group[f"block{block}_items"])
            block_values = group[f"block{block}_values"][:]
            if block_values.shape != (len(index), len(block_names)):
                raise ValueError("Reference block dimensions are inconsistent")
            for j, name in enumerate(block_names):
                if name in values:
                    raise ValueError(f"Duplicate reference column: {name}")
                values[name] = block_values[:, j]
        if set(values) != set(columns):
            raise ValueError("Reference columns disagree with stored blocks")
    units = integer_ids(values["unit"], "reference unit")
    hi = np.asarray(values["HI"], dtype=np.float64)
    if not np.isfinite(hi).all():
        raise ValueError("Nonfinite reference HI")
    expected = {}
    for f in flights:
        expected.setdefault(f["unit"], []).append(f["cycle"])
    if set(expected) != set(units.tolist()):
        raise ValueError("Reference unit set differs from flight unit set")
    cycles = np.empty(len(units), dtype=np.int64)
    for unit, raw_cycles in expected.items():
        rows = np.flatnonzero(units == unit)
        if len(rows) != len(raw_cycles) or not (np.diff(rows) == 1).all():
            raise ValueError(f"Reference count/order mismatch: unit={unit}")
        if raw_cycles != list(range(1, len(rows) + 1)):
            raise ValueError(f"Cannot infer reference cycles reliably: unit={unit}")
        cycles[rows] = np.arange(1, len(rows) + 1)
    frame = pd.DataFrame({"unit": units, "cycle": cycles, "hi_gt": hi})
    if frame.duplicated(["unit", "cycle"]).any() or len(frame) != len(flights):
        raise ValueError("Reference keys are not one-to-one with flights")
    return frame


def print_summary(data):
    for split in ("dev", "test"):
        flights = data[split]
        lengths = [len(f["x"]) for f in flights]
        print(f"{split}: units={sorted({f['unit'] for f in flights})}, flights={len(flights)}, "
              f"lengths={min(lengths)}..{max(lengths)}, dimensions=14/4")
        for flight in flights[:3]:
            print(f"  unit={flight['unit']} cycle={flight['cycle']} length={len(flight['x'])} "
                  f"raw_rows={flight['raw_stop'] - flight['raw_start']} first_sample_row={flight['first_sample_row']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepare", action="store_true", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    data = prepare_data(config)
    path = Path(config["data"]["processed_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    print_summary(data)
    print(f"Prepared: {path} (no HI or RUL loaded)")


if __name__ == "__main__":
    main()
