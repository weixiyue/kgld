"""Flight statistics versus frozen mu_I: five-seed, label-free ranking diagnostics."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import FlightDataset, collate_fn, load_config
from diagnose import apply_transform, fit_transform, pair_schedule
from diagnostics_latent import check_unchanged, load_inputs, new_output
from losses import hi_loss, order_loss
from metrics import spearman
from model import HIHead
from train import extract_training_means, representation_state, valid_pair_pools, validate_hi_config


def statistics(dataset):
    sensors, conditions = [], []
    for flight in dataset.flights:
        batch = collate_fn([flight])
        x, w = batch['x'][batch['mask']].double(), batch['w'][batch['mask']].double()
        if not torch.isfinite(x).all() or not torch.isfinite(w).all():
            raise ValueError('Nonfinite flight input')
        f1 = torch.cat((x.mean(0), x.std(0, unbiased=False)))
        quantiles = torch.quantile(x, torch.tensor([.1, .5, .9], dtype=torch.float64), dim=0, interpolation='linear')
        sensors.append(torch.cat((f1, quantiles.flatten())))
        conditions.append(torch.cat((w.mean(0), w.std(0, unbiased=False))))
    return torch.stack(sensors), torch.stack(conditions)


def ranking_rows(name, split, seed, scores, units, cycles, gap, kappa):
    """Same exhaustive r-based ranking definition as D1/D2, without labels."""
    rows = []
    for unit, pairs in valid_pair_pools(units, cycles, gap):
        indices = torch.where(units == unit)[0]
        early, late = scores[pairs[:, 0]], scores[pairs[:, 1]]
        # Check the enumerator independently in NumPy, including all ties.
        c = cycles[indices].cpu().numpy()
        expected = (c[None, :] - c[:, None] >= gap)
        assert len(pairs) == int(expected.sum())
        r = scores[indices].cpu().numpy().astype(np.float64)
        differences = (r[:, None] - r[None, :])[expected]
        correct, ties = int((late > early).sum()), int((late == early).sum())
        assert correct == int((differences < 0).sum()) and ties == int((differences == 0).sum())
        order = order_loss(early, late, kappa).item()
        np.testing.assert_allclose(order, np.logaddexp(0, differences / kappa).mean(), rtol=1e-6, atol=1e-7)
        rows.append({'feature_set': name, 'split': split, 'seed': seed, 'unit': str(int(unit)),
                     'flights': len(indices), 'pairs': len(pairs), 'correct_count': correct, 'tie_count': ties,
                     'accuracy': correct / len(pairs), 'order': order,
                     'spearman_hi_cycle': spearman(torch.sigmoid(-scores[indices]).cpu().numpy(), c)})
    if len(rows) != len(torch.unique(units)):
        raise ValueError('A unit has no valid pairs')
    frame = pd.DataFrame(rows)
    rows.append({'feature_set': name, 'split': split, 'seed': seed, 'unit': 'all',
                 **{key: int(frame[key].sum()) for key in ('flights', 'pairs', 'correct_count', 'tie_count')},
                 **{key: float(frame[key].mean()) for key in ('accuracy', 'order', 'spearman_hi_cycle')}})
    return rows


def run(config, representation, smoke):
    validate_hi_config(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        torch.set_num_threads(1)
    digest = hashlib.sha256(Path(representation).read_bytes()).hexdigest()
    model, processed = load_inputs(config, representation, device)
    before = representation_state(model)
    output = new_output('features')
    mode, epochs = ('smoke', 3) if smoke else ('full', 100)
    print(f'features {mode}: device={device}; epochs={epochs}; output={output}', flush=True)
    data, sensor_data, condition_data, keys = {}, {}, {}, {}
    for split in ('dev', 'test'):
        dataset = FlightDataset(processed, split)
        if smoke:
            selected = []
            for unit in sorted({f['unit'] for f in dataset.flights}):
                flights = sorted((f for f in dataset.flights if f['unit'] == unit), key=lambda f: f['cycle'])
                indices = np.unique(np.linspace(0, len(flights) - 1, min(8, len(flights))).round().astype(int))
                selected.extend(flights[i] for i in indices)
            dataset.flights = selected
        data[split] = extract_training_means(model, dataset, 16, device)
        sensor_data[split], condition_data[split] = statistics(dataset)
        keys[split] = [[f['unit'], f['cycle']] for f in dataset.flights]
        assert list(map(list, zip(data[split][1].cpu().tolist(), data[split][2].cpu().tolist()))) == keys[split]

    # Intercept plus eight W statistics; fit solely on dev in float64.
    design = {split: torch.cat((torch.ones(len(w), 1, dtype=torch.float64), w), dim=1).numpy()
              for split, w in condition_data.items()}
    coefficients, _, rank, singular = np.linalg.lstsq(design['dev'], sensor_data['dev'][:, :28].numpy(), rcond=None)
    residuals = {split: torch.from_numpy(sensor_data[split][:, :28].numpy() - design[split] @ coefficients)
                 for split in data}
    np.testing.assert_allclose(design['dev'].T @ residuals['dev'].numpy(), 0, atol=1e-8, rtol=0)
    raw = {'F0': {split: values[0] for split, values in data.items()},
           'F1': {split: values[:, :28].float().to(device) for split, values in sensor_data.items()},
           'F2': {split: values.float().to(device) for split, values in sensor_data.items()},
           'F3': {split: values.float().to(device) for split, values in residuals.items()}}
    transforms = {name: fit_transform(values['dev']) for name, values in raw.items()}
    inputs = {name: {split: apply_transform(values, transforms[name]) for split, values in splits.items()}
              for name, splits in raw.items()}
    for name in raw:
        dev = raw[name]['dev'].double().cpu().numpy()
        np.testing.assert_allclose(transforms[name]['mean'], dev.mean(0), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(transforms[name]['std'], dev.std(0, ddof=0), rtol=1e-12, atol=1e-12)
        for split in data:
            assert torch.isfinite(inputs[name][split]).all()
            expected = ((raw[name][split].double().cpu() - transforms[name]['mean']) / transforms[name]['denominator']).float()
            torch.testing.assert_close(inputs[name][split].cpu(), expected, rtol=0, atol=0)

    feature_names = [f'{sensor}_{stat}' for stat in ('mean', 'std') for sensor in processed['columns']['x']]
    correlations = []
    for split, values in sensor_data.items():
        units, cycles = data[split][1].cpu().numpy(), data[split][2].cpu().numpy()
        for dimension, name in enumerate(feature_names):
            rhos = np.array([spearman(values[units == u, dimension].numpy(), cycles[units == u]) for u in np.unique(units)])
            avg = float(rhos.mean())
            correlations.append({'split': split, 'feature': name, 'mean_spearman': avg,
                                 'same_sign_units': int((rhos * avg > 0).sum()),
                                 'positive_units': int((rhos > 0).sum()), 'negative_units': int((rhos < 0).sum()),
                                 'zero_units': int((rhos == 0).sum()), 'units': len(rhos), 'mode': mode})
    pd.DataFrame(correlations).to_csv(output / 'feature_spearman.csv', index=False, float_format='%.17g')

    settings = config['training']['hi']
    units, cycles = data['dev'][1:]
    pools = valid_pair_pools(units, cycles, settings['min_cycle_gap'])
    results, logs, predictions, saved_heads, schedules = [], [], [], {}, {}
    for seed in range(42, 47):
        schedule = pair_schedule(pools, len(units), epochs, seed)
        schedules[str(seed)] = schedule.cpu()
        for name, values in inputs.items():
            # Exactly D1's seed-reset CPU initialization, then transfer to device.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                head = HIHead(values['dev'].shape[1], 16).to(device)
            initial = copy.deepcopy(head.state_dict())
            optimizer = torch.optim.Adam(head.parameters(), lr=settings['learning_rate'], weight_decay=0)
            for epoch, pairs in enumerate(schedule, 1):
                totals = dict.fromkeys(('loss', 'order', 'l2'), 0.0)
                for start in range(0, len(pairs), 128):
                    batch = pairs[start:start + 128]
                    optimizer.zero_grad(set_to_none=True)
                    losses = hi_loss(head(values['dev'][batch[:, 0]]), head(values['dev'][batch[:, 1]]), head,
                                     settings['kappa'], settings['l2_coefficient'])
                    if not all(torch.isfinite(v) for v in losses.values()):
                        raise RuntimeError('Nonfinite head loss')
                    losses['loss'].backward()
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
                    optimizer.step()
                    for key in totals:
                        totals[key] += losses[key].item() * len(batch)
                logs.append({'feature_set': name, 'seed': seed, 'epoch': epoch, 'pairs': len(pairs),
                             **{k: v / len(pairs) for k, v in totals.items()}, 'mode': mode})
            assert any(not torch.equal(initial[k], v) for k, v in head.state_dict().items())
            head.eval()
            state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            saved_heads[f'{name}_{seed}'] = {'input_dim': values['dev'].shape[1], 'initial': {k: v.cpu() for k, v in initial.items()}, 'state_dict': state}
            with torch.no_grad():
                for split in data:
                    scores = head(values[split])
                    results.extend(ranking_rows(name, split, seed, scores, data[split][1], data[split][2], 5, 1))
                    predictions.extend({'feature_set': name, 'seed': seed, 'split': split, 'unit': u, 'cycle': c,
                                        'r': float(r), 'hi_raw': float(hi), 'mode': mode}
                                       for (u, c), r, hi in zip(keys[split], scores.cpu(), torch.sigmoid(-scores).cpu()))
            print(f'{name} seed={seed}: finished {epochs} epochs', flush=True)

    check_unchanged(model, before, representation, digest)
    frame = pd.DataFrame(results).assign(mode=mode)
    summary = []
    for (name, split), group in frame[frame.unit == 'all'].groupby(['feature_set', 'split']):
        summary.append({'feature_set': name, 'split': split, 'seeds': len(group), 'mode': mode,
                        **{f'{metric}_{stat}': float(getattr(group[metric], stat)(**({'ddof': 0} if stat == 'std' else {})))
                           for metric in ('accuracy', 'order', 'spearman_hi_cycle') for stat in ('mean', 'std')}})
    summary = pd.DataFrame(summary)
    prediction_table = pd.DataFrame(predictions)
    frame.to_csv(output / 'ranking.csv', index=False, float_format='%.17g')
    summary.to_csv(output / 'seed_summary.csv', index=False, float_format='%.17g')
    pd.DataFrame(logs).to_csv(output / 'losses.csv', index=False, float_format='%.17g')
    prediction_table.to_csv(output / 'predictions.csv', index=False, float_format='%.17g')
    torch.save({'heads': saved_heads, 'transforms': transforms, 'regression_coefficients': torch.from_numpy(coefficients),
                'schedules': schedules, 'keys': keys, 'config': config, 'mode': mode, 'source_sha256': digest,
                'source': str(representation)}, output / 'diagnostic_state.pt')
    saved = torch.load(output / 'diagnostic_state.pt', map_location='cpu', weights_only=True)
    for key, head_info in saved['heads'].items():
        name, seed = key.split('_')
        head = HIHead(head_info['input_dim'], 16).to(device).eval()
        head.load_state_dict(head_info['state_dict'])
        for split in data:
            parameters = saved['transforms'][name]
            if name == 'F3':
                restored_raw = torch.from_numpy(sensor_data[split][:, :28].numpy() - design[split] @ saved['regression_coefficients'].numpy()).float().to(device)
            else:
                restored_raw = raw[name][split]
            with torch.no_grad():
                restored = head(apply_transform(restored_raw, parameters)).cpu().numpy()
            subset = prediction_table[(prediction_table.feature_set == name) & (prediction_table.seed == int(seed)) & (prediction_table.split == split)]
            np.testing.assert_array_equal(restored, subset.r.to_numpy())

    notes = [f'Mode={mode}; device={device}; seeds=42..46; epochs={epochs}; final epoch only.',
             'No HI ground truth or RUL loaded; cycle only constructs pairs/evaluation.',
             'Flight std and dev standardization use ddof=0; linear quantiles; no test clipping.',
             f'F3: dev-only OLS with intercept; design rank={rank}/9; singular values={singular.tolist()}.',
             'Accuracy/order/signed HI-cycle Spearman weight units equally; ties are incorrect.',
             'Across-seed standard deviations use ddof=0. Different input dimensions change head capacity.',
             'Dev ranking is an in-sample training diagnostic; test provides held-out-unit evidence.']
    for name, parameters in transforms.items():
        notes.append(f'{name}: zero-std dimensions={parameters["zero_std_dimensions"]}; head parameters={16 * (inputs[name]["dev"].shape[1] + 1) + 17}.')
    f0 = frame[(frame.feature_set == 'F0') & (frame.seed == 42) & (frame.unit == 'all')]
    notes.append('F0 seed=42 reference (same 20261010 checkpoint and CUDA): dev=58.26%, test=55.99%; smoke is not a reproduction check.')
    for _, row in f0.iterrows():
        notes.append(f'F0 seed=42 {row["split"]}: {100 * row.accuracy:.6f}%; order={row.order:.9f}; signed Spearman={row.spearman_hi_cycle:.9f}.')
    (output / 'summary.txt').write_text('\n'.join(notes) + '\n\n' + summary.to_string(index=False) + '\n', encoding='utf-8')
    (output / 'verification.json').write_text(json.dumps({'mode': mode, 'device': str(device), 'source_sha256': digest,
        'frozen_and_source_unchanged': True, 'fit_split': 'dev', 'std_ddof': 0, 'regression_intercept': True,
        'regression_rank': int(rank), 'shared_pair_schedule_per_seed': True, 'reload_predictions_exact': True,
        'exhaustive_pairs_independently_checked': True, 'hi_rul_read': False, 'keys': keys}, indent=2), encoding='utf-8')
    print(summary.to_string(index=False), flush=True)
    print(f'PASS: features {mode}; output={output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--representation', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(load_config(args.config), args.representation, args.smoke)


if __name__ == '__main__':
    main()
