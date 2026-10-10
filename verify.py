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


def verify_model(config):
    from losses import (diagonal_gaussian_kl, head_l2, hi_loss, order_loss,
                        reconstruction_loss, representation_loss)
    from model import HIHead, OCALD, time_positions

    torch.manual_seed(config["training"]["seed"])
    if config["training"]["device"] != "auto":
        raise ValueError("Unapproved device selection; expected auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(1)  # Small validation batches avoid CPU thread overhead.
    data = load_processed(config)
    dataset = FlightDataset(data, "dev")
    longest = max(range(len(dataset)), key=lambda index: len(dataset[index]["x"]))
    indices = list(dict.fromkeys([0, 1, longest]))
    batch = {key: value.to(device) for key, value in collate_fn([dataset[i] for i in indices]).items()}
    x, w, mask = batch["x"], batch["w"], batch["mask"]
    model = OCALD(config).to(device)
    output = model(x, w, mask)
    for key in ("mu_i", "logvar_i", "mu_c", "logvar_c", "prior_mu_c", "prior_logvar_c"):
        check(output[key].shape == (len(indices), 8) and torch.isfinite(output[key]).all(), f"Invalid {key}")
    check(output["x_hat"].shape == x.shape and torch.isfinite(output["x_hat"]).all(), "Invalid reconstruction")
    losses = representation_loss(x, mask, output, config["training"]["representation"]["sigma_x_squared"])
    check(all(torch.isfinite(value) for value in losses.values()), "Nonfinite real-batch loss")
    losses["loss"].backward()
    for name in ("sensor_encoder", "conditional_prior", "decoder"):
        parameters = list(getattr(model, name).parameters())
        check(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters), f"Missing/nonfinite gradients: {name}")
        check(sum(p.grad.abs().sum().item() for p in parameters) > 0, f"Zero gradient: {name}")
    check(all(p.grad is None for p in model.hi_head.parameters()), "Stage 1 objective reached HI head")
    print(f"Device: {device}; real dev batch indices={indices}, lengths={mask.sum(1).tolist()}, X={tuple(x.shape)}, W={tuple(w.shape)}")
    print("Parameter counts:", {name: sum(p.numel() for p in getattr(model, name).parameters())
                                for name in ("sensor_encoder", "conditional_prior", "decoder", "hi_head")})
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
    print("Random-initialization loss:", {key: value.item() for key, value in losses.items()})

    # Compare to an independent PyTorch distribution implementation in float64.
    mu_q, logvar_q, mu_p, logvar_p = [torch.randn(5, 8, dtype=torch.float64, device=device) for _ in range(4)]
    actual = diagonal_gaussian_kl(mu_q, logvar_q, mu_p, logvar_p)
    q = torch.distributions.Normal(mu_q, (0.5 * logvar_q).exp())
    p = torch.distributions.Normal(mu_p, (0.5 * logvar_p).exp())
    expected = torch.distributions.kl_divergence(q, p).sum(-1)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(diagonal_gaussian_kl(mu_q, logvar_q, mu_q, logvar_q), torch.zeros_like(actual), rtol=0, atol=1e-12)
    print(f"KL: PyTorch agreement, max absolute error={(actual - expected).abs().max().item():.3g}; q=p gives zero")

    # Hand calculation: valid residual=1, reconstruction=[2,1], KL_I=[0.5,2].
    toy_mask = torch.tensor([[True, True], [True, False]], device=device)
    toy_x = torch.ones(2, 2, 2, device=device).masked_fill(~toy_mask[:, :, None], float("nan"))
    zero = torch.zeros(2, 1, device=device)
    toy_output = {"x_hat": torch.zeros_like(toy_x), "mu_i": torch.tensor([[1.], [2.]], device=device),
                  "logvar_i": zero, "mu_c": zero, "logvar_c": zero, "prior_mu_c": zero, "prior_logvar_c": zero}
    torch.testing.assert_close(reconstruction_loss(toy_x, toy_output["x_hat"], toy_mask, 1.0),
                               torch.tensor([2., 1.], device=device))
    torch.testing.assert_close(representation_loss(toy_x, toy_mask, toy_output, 1.0)["loss"],
                               torch.tensor(2.75, device=device))
    print("ELBO: hand-calculated flight-summed reconstruction and flight-averaged total=2.75 verified")

    model.eval()
    with torch.no_grad():
        posterior = model.encode(x, mask)
        prior = model.conditional_prior(w, mask)
        hi = model.predict_hi(x, mask)
        torch.testing.assert_close(hi, torch.sigmoid(-model.hi_head(posterior[0])), rtol=0, atol=0)
        torch.testing.assert_close(hi, model.predict_hi(x, mask), rtol=0, atol=0)
        check(hi.shape == (len(indices),) and ((hi > 0) & (hi < 1)).all(), "Invalid deterministic HI")
        changed_x = x.masked_fill(~mask[:, :, None], float("nan"))
        changed_w = w.masked_fill(~mask[:, :, None], float("nan"))
        padded_mask = torch.nn.functional.pad(mask, (0, 11), value=False)
        padded_x = torch.nn.functional.pad(changed_x, (0, 0, 0, 11), value=float("nan"))
        padded_w = torch.nn.functional.pad(changed_w, (0, 0, 0, 11), value=float("nan"))
        for candidate_x, candidate_w, candidate_mask in ((changed_x, changed_w, mask), (padded_x, padded_w, padded_mask)):
            for base, changed in zip(posterior, model.encode(candidate_x, candidate_mask)):
                torch.testing.assert_close(base, changed, rtol=1e-6, atol=1e-6)
            for base, changed in zip(prior, model.conditional_prior(candidate_w, candidate_mask)):
                torch.testing.assert_close(base, changed, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(hi, model.predict_hi(candidate_x, candidate_mask), rtol=1e-6, atol=1e-6)
            base_reconstruction = model.decode(posterior[0], posterior[2], time_positions(mask))
            changed_reconstruction = model.decode(posterior[0], posterior[2], time_positions(candidate_mask))[:, :mask.shape[1]]
            torch.testing.assert_close(base_reconstruction[mask], changed_reconstruction[mask], rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(reconstruction_loss(x, base_reconstruction, mask, 1.0),
                                       reconstruction_loss(candidate_x, model.decode(posterior[0], posterior[2], time_positions(candidate_mask)), candidate_mask, 1.0))
        for j, index in enumerate(indices):
            single = {key: value.to(device) for key, value in collate_fn([dataset[index]]).items()}
            for base, unpadded in zip(posterior, model.encode(single["x"], single["mask"])):
                torch.testing.assert_close(base[j:j + 1], unpadded, rtol=1e-6, atol=1e-6)
        positions = time_positions(mask)
        torch.testing.assert_close(positions[:, 0], torch.zeros(len(indices), device=device))
        torch.testing.assert_close(positions.gather(1, (mask.sum(1) - 1)[:, None]).squeeze(1), torch.ones(len(indices), device=device))
        torch.testing.assert_close(time_positions(torch.tensor([[True, False]], device=device)), torch.zeros(1, 2, device=device))

    # Forward hooks establish that altered W changes only the prior path. Fix
    # sampling seed so a stochastic latent draw cannot obscure the comparison.
    seen = []
    hook = model.sensor_encoder.register_forward_hook(lambda module, inputs, result: seen.append(result))
    try:
        with torch.no_grad():
            torch.manual_seed(42)
            base = model(x, w, mask)
            torch.manual_seed(42)
            altered = model(x, w + 0.75, mask)
        for first, second in zip(seen[0], seen[1]):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        check(not torch.equal(base["prior_mu_c"], altered["prior_mu_c"]), "W did not reach conditional prior")
        torch.testing.assert_close(base["x_hat"], altered["x_hat"], rtol=0, atol=0)
        with torch.no_grad():
            torch.testing.assert_close(hi, model.predict_hi(x, mask), rtol=0, atol=0)
    finally:
        hook.remove()
    print("Mask/time: padding replacement/extension, single-flight comparison and length-one position verified")
    print("Information paths: W affects prior; posterior/decoder/HI independence and deterministic sigmoid(-r) verified")

    early = torch.tensor([-1., 0.], device=device, requires_grad=True)
    late = torch.tensor([1., 2.], device=device, requires_grad=True)
    kappa = config["training"]["hi"]["kappa"]
    correct, reversed_loss = order_loss(early, late, kappa), order_loss(late, early, kappa)
    check(correct < reversed_loss, "Reversed order has lower loss")
    correct.backward()
    check((early.grad > 0).all() and (late.grad < 0).all(), "Wrong ordering gradient direction")
    head = HIHead(8, 16).to(device)
    with torch.no_grad():
        for name, parameter in head.named_parameters():
            parameter.fill_(1 if name.endswith("bias") else 2)
    # 144 weights of value 2 plus 17 biases of value 1: 144*4+17=593.
    torch.testing.assert_close(head_l2(head), torch.tensor(593., device=device))
    ordinal = hi_loss(early.detach(), late.detach(), head, kappa, config["training"]["hi"]["l2_coefficient"])
    torch.testing.assert_close(ordinal["loss"], correct.detach() + 0.0001 * 593)
    ordinal["loss"].backward()
    check(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()), "Missing head L2 gradients")
    print(f"Order: correct={correct.item():.6f}, reversed={reversed_loss.item():.6f}; gradient direction and bias-inclusive L2 verified")
    print("PASS: T2 model and loss verification (real forward/backward plus analytical checks; no training)")


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
    parser.add_argument("--task", required=True, choices=["data", "data-plot", "model"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["dev", "test"])
    parser.add_argument("--unit", type=int)
    parser.add_argument("--cycle", type=int)
    parser.add_argument("--parameters", nargs="+", help="X_s/W variable names to plot")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.task == "data":
        verify_data(config)
    elif args.task == "model":
        verify_model(config)
    else:
        if any(value is None for value in (args.split, args.unit, args.cycle, args.parameters)):
            parser.error("data-plot requires --split, --unit, --cycle and --parameters")
        plot_data(config, args.split, args.unit, args.cycle, args.parameters)


if __name__ == "__main__":
    main()
