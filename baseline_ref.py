"""Paper-style deterministic convolutional AE, separate from formal OCALD."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from data import FlightDataset, collate_fn, load_config, load_processed, read_reference, read_rul
from losses import order_loss
from metrics import evaluate_metrics, spearman
from train import valid_pair_pools


VARIANTS = ('V0', 'V_NG', 'V_C', 'V_ORD')


class ReferenceAE(nn.Module):
    def __init__(self, max_length):
        super().__init__()
        self.max_length = max_length
        self.encoder = nn.Sequential(
            nn.Conv1d(14, 128, 11, padding=5), nn.ReLU(),
            nn.Conv1d(128, 64, 11, padding=5), nn.ReLU(),
            nn.Conv1d(64, 16, 11, padding=5), nn.ReLU())
        self.latent = nn.Linear(16 * max_length, 1)
        self.decoder = nn.Sequential(
            nn.Conv1d(5, 16, 11, padding=5), nn.ReLU(),
            nn.Conv1d(16, 64, 11, padding=5), nn.ReLU(),
            nn.Conv1d(64, 128, 11, padding=5), nn.ReLU())
        self.output = nn.Linear(128, 14)

    def encode(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.max_length, 14):
            raise ValueError('Encoder expects X [B, actual_global_T_max, 14]')
        return self.latent(self.encoder(x.transpose(1, 2)).flatten(1)).squeeze(-1)

    def decode(self, z, w):
        if z.ndim != 1 or w.shape != (len(z), self.max_length, 4):
            raise ValueError('Decoder expects scalar Z and aligned W')
        inputs = torch.cat((z[:, None, None].expand(-1, self.max_length, 1), w), dim=2)
        return self.output(self.decoder(inputs.transpose(1, 2)).transpose(1, 2))

    def forward(self, x, w):
        z = self.encode(x)
        return z, self.decode(z, w)


def fixed_batch(flights, max_length, device):
    batch = collate_fn(flights)
    amount = max_length - batch['mask'].shape[1]
    if amount < 0:
        raise ValueError('Flight exceeds checkpoint T_max; truncation prohibited')
    for field in ('x', 'w'):
        batch[field] = F.pad(batch[field], (0, 0, 0, amount))
    batch['mask'] = F.pad(batch['mask'], (0, amount), value=False)
    return {key: value.to(device) for key, value in batch.items()}


def masked_mae(x, prediction, mask):
    if x.shape != prediction.shape or mask.shape != x.shape[:2] or mask.dtype != torch.bool:
        raise ValueError('Invalid reconstruction/mask shape')
    if (mask.sum(1) == 0).any():
        raise ValueError('Empty flight')
    residual = x.masked_fill(~mask[:, :, None], 0) - prediction.masked_fill(~mask[:, :, None], 0)
    return (residual.abs().sum((1, 2)) / (mask.sum(1) * x.shape[2])).mean()


def constraint_loss(z, units, cycles, variant):
    if variant == 'V0':
        return z.sum() * 0, 0
    if variant == 'V_C':
        correlations, zero_groups = [], 0
        for unit in torch.unique(units):
            indices = units == unit
            values, time = z[indices], cycles[indices].to(z.dtype)
            if len(values) < 2 or torch.unique(time).numel() < 2:
                raise ValueError('Pearson needs distinct cycles per unit')
            values, time = values - values.mean(), time - time.mean()
            # vector_norm has a defined zero derivative, unlike sqrt(sum(z**2)).
            norm = torch.linalg.vector_norm(values)
            zero_groups += int(norm.detach() == 0)
            correlations.append((values * time).sum() / norm.clamp_min(1e-8) / torch.linalg.vector_norm(time))
        return torch.stack(correlations).mean(), zero_groups
    pairs = torch.cat([pairs for _, pairs in valid_pair_pools(units, cycles, 5)])
    early, late = z[pairs[:, 0]], z[pairs[:, 1]]
    if variant == 'V_NG':
        return (late - early).relu().mean(), 0
    if variant == 'V_ORD':
        return order_loss(-early, -late, 1.0), 0
    raise ValueError(f'Unknown variant: {variant}')


def select_datasets(processed, smoke):
    datasets = {}
    for split in ('dev', 'test'):
        dataset = FlightDataset(processed, split)
        expected = list(range(1, 10)) if split == 'dev' else list(range(10, 16))
        if sorted({f['unit'] for f in dataset.flights}) != expected:
            raise ValueError(f'Unexpected {split} units')
        keys = [(f['unit'], f['cycle']) for f in dataset.flights]
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate flight keys')
        if smoke:
            chosen = []
            for unit in expected:
                flights = sorted((f for f in dataset.flights if f['unit'] == unit), key=lambda f: f['cycle'])
                indices = np.unique(np.linspace(0, len(flights) - 1, min(8, len(flights))).round().astype(int))
                chosen.extend(flights[i] for i in indices)
            dataset.flights = chosen
        datasets[split] = dataset
    return datasets


def make_schedule(flights, epochs, batches, seed):
    units = torch.tensor([f['unit'] for f in flights])
    cycles = torch.tensor([f['cycle'] for f in flights])
    groups = [torch.where(units == u)[0] for u in torch.unique(units)]
    if len(groups) < 4 or any(len(g) < 5 or cycles[g].max() - cycles[g].min() < 5 for g in groups):
        raise ValueError('Need >=4 units, each with >=5 flights and a valid gap')
    rng = torch.Generator().manual_seed(seed)
    schedule = []
    for _ in range(epochs * batches):
        selections = []
        for group_id in torch.randperm(len(groups), generator=rng)[:4]:
            group = groups[int(group_id)]
            while True:
                selected = group[torch.randperm(len(group), generator=rng)[:5]]
                if cycles[selected].max() - cycles[selected].min() >= 5:
                    selections.append(selected)
                    break
        batch = torch.cat(selections)
        assert len(torch.unique(batch)) == 20
        assert torch.unique(units[batch], return_counts=True)[1].tolist() == [5] * 4
        pairs = torch.cat([p for _, p in valid_pair_pools(units[batch], cycles[batch], 5)])
        assert (units[batch][pairs[:, 0]] == units[batch][pairs[:, 1]]).all()
        assert (cycles[batch][pairs[:, 1]] - cycles[batch][pairs[:, 0]] >= 5).all()
        schedule.append(batch)
    return torch.stack(schedule).reshape(epochs, batches, 20)


def check_mechanisms(max_length, probe, device):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(123)
        model = ReferenceAE(max_length).to(device)
    with torch.no_grad():
        z, prediction = model(probe['x'], probe['w'])
        assert z.shape == (len(probe['x']),) and prediction.shape == probe['x'].shape
        altered_w = probe['w'] + 1
        # X-only encoder: W affects only reconstruction.
        torch.testing.assert_close(model.encode(probe['x']), z, rtol=0, atol=0)
        assert not torch.equal(model.decode(z, altered_w), prediction)
    x = torch.tensor([[[1.], [3.], [float('nan')]], [[2.], [float('nan')], [float('nan')]]], device=device)
    mask = torch.tensor([[True, True, False], [True, False, False]], device=device)
    predicted = torch.zeros_like(x, requires_grad=True)
    loss = masked_mae(x, predicted, mask)
    torch.testing.assert_close(loss, torch.tensor(2., device=device))
    loss.backward()
    assert (predicted.grad[~mask] == 0).all()
    predicted2 = predicted.detach().clone().masked_fill(~mask[:, :, None], float('nan'))
    torch.testing.assert_close(masked_mae(x, predicted2, mask), loss.detach())
    units = torch.tensor([1, 1, 1, 2, 2, 2], device=device)
    cycles = torch.tensor([1, 6, 11, 1, 6, 11], device=device)
    descending = torch.tensor([1., .5, 0., 1., .5, 0.], device=device)
    for variant in ('V_NG', 'V_C', 'V_ORD'):
        correct, _ = constraint_loss(descending, units, cycles, variant)
        reverse, _ = constraint_loss(-descending, units, cycles, variant)
        assert correct < reverse
        model.zero_grad(set_to_none=True)
        z, reconstruction = model(probe['x'], probe['w'])
        penalty, _ = constraint_loss(z, probe['unit'], probe['cycle'], variant)
        penalty.backward(retain_graph=True)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.encoder.parameters())
        assert any(torch.count_nonzero(p.grad) for p in model.encoder.parameters())
        assert all(p.grad is None for p in model.decoder.parameters())
        model.zero_grad(set_to_none=True)
        (masked_mae(probe['x'], reconstruction, probe['mask']) + penalty).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    constant = torch.zeros(6, device=device, requires_grad=True)
    correlation, zero_groups = constraint_loss(constant, units, cycles, 'V_C')
    correlation.backward()
    assert zero_groups == 2 and correlation == 0 and torch.isfinite(constant.grad).all()
    # Direction and scaling checks include unclipped test values.
    scaled = (descending - descending.min()) / (descending.max() - descending.min())
    assert (scaled[0] > scaled[2]) and ((torch.tensor([-1., 2.]) - 0) / 1).tolist() == [-1., 2.]
    print('PASS: shape, X-only path, masked MAE, pairs, constraint gradients and HI direction', flush=True)


def predict_z(model, dataset, device):
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), 20):
            batch = fixed_batch(dataset.flights[start:start + 20], model.max_length, device)
            values.append(model.encode(batch['x']).cpu())
    result = torch.cat(values)
    if not torch.isfinite(result).all():
        raise RuntimeError('Nonfinite Z prediction')
    return result


def ranking(table):
    units = torch.tensor(table.unit.to_numpy())
    cycles = torch.tensor(table.cycle.to_numpy())
    z = torch.tensor(table.z.to_numpy(), dtype=torch.float64)
    hi = torch.tensor(table.hi_scaled.to_numpy(), dtype=torch.float64)
    rows = []
    for unit, pairs in valid_pair_pools(units, cycles, 5):
        indices = torch.where(units == unit)[0]
        early, late = pairs[:, 0], pairs[:, 1]
        correct, ties = int((z[early] > z[late]).sum()), int((z[early] == z[late]).sum())
        available = torch.isfinite(hi[indices]).all().item()
        # An independent cycle-difference matrix checks pair enumeration.
        c, values = cycles[indices].numpy(), z[indices].numpy()
        mask = c[None, :] - c[:, None] >= 5
        delta = (values[None, :] - values[:, None])[mask]
        assert int(mask.sum()) == len(pairs) and int((delta < 0).sum()) == correct and int((delta == 0).sum()) == ties
        row = {'unit': str(int(unit)), 'flights': len(indices), 'pairs': len(pairs), 'correct_count': correct,
               'tie_count': ties, 'accuracy': correct / len(pairs), 'tie_rate': ties / len(pairs),
               'order_z': order_loss(-z[early], -z[late], 1).item(),
               'order_hi': order_loss(-hi[early], -hi[late], 1).item() if available else None,
               'spearman_z_cycle': spearman(values, c),
               'spearman_hi_cycle': spearman(hi[indices].numpy(), c) if available else None}
        if available:
            assert int((hi[early] > hi[late]).sum()) == correct and int((hi[early] == hi[late]).sum()) == ties
            np.testing.assert_allclose(row['order_hi'], np.logaddexp(0, (hi[late] - hi[early]).numpy()).mean(), atol=1e-14)
        rows.append(row)
    if len(rows) != torch.unique(units).numel():
        raise ValueError('Unit missing valid evaluation pairs')
    overall = {'unit': 'all', **{key: sum(row[key] for row in rows) for key in ('flights', 'pairs', 'correct_count', 'tie_count')}}
    for key in ('accuracy', 'tie_rate', 'order_z', 'order_hi', 'spearman_z_cycle', 'spearman_hi_cycle'):
        overall[key] = float(np.mean([row[key] for row in rows])) if all(row[key] is not None for row in rows) else None
    return rows + [overall]


def finite_report(value):
    if isinstance(value, dict):
        result = {key: finite_report(item) for key, item in value.items()}
        if 'value' in value and value['value'] is not None and not np.isfinite(value['value']):
            result['value'] = None
            result['reason'] = 'Reference metric formula produced a nonfinite result; unavailable, no clipping'
        return result
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def metric_table(table):
    if table.hi_scaled.isna().all():
        return {'available': False, 'reason': 'Dev Z range is zero'}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = evaluate_metrics(table.hi_scaled, table.hi_gt, table.unit, table.cycle, table.rul, 42)
    return {'available': True, **finite_report(result), 'warnings': sorted({str(w.message) for w in caught})}


def plot_split(table, report, path, variant, epochs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(math.ceil(table.unit.nunique() / 2), 2, figsize=(14, 2.8 * math.ceil(table.unit.nunique() / 2)),
                                layout='constrained', squeeze=False)
    figure.suptitle(f'DS03 {table.split.iloc[0]} | {variant} | seed 42 | {epochs} epochs | dev-scaled HI')
    for axis, (unit, group) in zip(axes.flat, table.groupby('unit', sort=True)):
        group = group.sort_values('cycle')
        axis.plot(group.cycle, group.hi_gt, color='black', linewidth=1, label='Reference HI')
        if report['available']:
            line, = axis.plot(group.cycle, group.hi_scaled, linewidth=1.1, label='Predicted HI (scaled)')
            assert np.array_equal(line.get_xdata(), group.cycle) and np.array_equal(line.get_ydata(), group.hi_scaled)
            metric = report['per_unit'][str(unit)]
            details = ' '.join(f"{label}={metric[key]['value']:.3f}" if metric[key]['value'] is not None else f'{label}=N/A'
                               for label, key in [('MAE', 'mae'), ('MAPE%', 'mape'), ('M', 'monotonicity'), ('rho', 'spearman'), ('MI', 'mi')])
        else:
            details = 'Scaled HI unavailable: zero dev Z range'
        axis.set_title(f'unit {unit}\n{details}', fontsize=9)
        axis.set_xlabel('Flight cycle')
        axis.set_ylabel('HI')
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    for axis in list(axes.flat)[table.unit.nunique():]:
        axis.set_visible(False)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def protected_files(config):
    files = ['model.py', 'losses.py', 'train.py', 'metrics.py', 'evaluate.py', 'data.py', 'verify.py',
             'config.yaml', 'README.md', 'AGENTS.md', 'OCALD_Implementation_Plan.md', config['data']['processed_path']]
    notebook = Path('baseline/03_train_evaluate_proposals.ipynb')
    if notebook.exists():
        files.append(str(notebook))
    return {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in files}


def run(config, epochs, lr, seeds, smoke):
    processed = load_processed(config)
    hashes = protected_files(config)
    # Fixed shape is descriptive only: no fitting of test values.
    max_length = max(len(f['x']) for split in ('dev', 'test') for f in processed[split])
    datasets = select_datasets(processed, smoke)
    batches = 2 if smoke else math.ceil(len(datasets['dev']) / 20)
    epochs = 3 if smoke else epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        torch.set_num_threads(1)
    out = Path('outputs/baseline_ref') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    settings = {'mode': 'smoke' if smoke else 'full', 'epochs': epochs, 'lr': lr, 'seeds': seeds,
                'device': str(device), 'T_max': max_length, 'batches_per_epoch': batches,
                'batch_size': 20, 'kernel': 11, 'padding': 5, 'lambda': {'V0': 0, 'V_NG': 1, 'V_C': 1, 'V_ORD': 1},
                'mae': 'mean_of_per_flight_valid_element_means', 'correlation_epsilon': 1e-8, 'mi_seed': 42,
                'batch_scheme': '4 distinct units x 5 distinct flights; redraw group without valid gap>=5; replacement across batches',
                'order_hi': 'softplus(HI_late-HI_early), temperature=1', 'weight_decay': 0,
                'config': config, 'protected_sha256': hashes,
                'keys': {split: [[f['unit'], f['cycle']] for f in dataset.flights] for split, dataset in datasets.items()}}
    dump_json(out / 'run_settings.json', settings)
    print(f"baseline_ref {settings['mode']}: device={device}; T_max={max_length}; dev/test={len(datasets['dev'])}/{len(datasets['test'])}; output={out}", flush=True)
    schedules = {str(seed): make_schedule(datasets['dev'].flights, epochs, batches, seed) for seed in seeds}
    torch.save({'schedules': schedules, 'dev_keys': settings['keys']['dev']}, out / 'batch_schedule.pt')
    probe = fixed_batch([datasets['dev'].flights[i] for i in schedules[str(seeds[0])][0, 0].tolist()], max_length, device)
    check_mechanisms(max_length, probe, device)
    del probe
    for seed in seeds:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            initial = ReferenceAE(max_length).state_dict()
        initial_hash = hashlib.sha256(b''.join(value.numpy().tobytes() for value in initial.values())).hexdigest()
        for variant in VARIANTS:
            folder = out / variant / f'seed_{seed}'
            folder.mkdir(parents=True)
            model = ReferenceAE(max_length).to(device)
            model.load_state_dict(initial)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
            logs = []
            for epoch, selections in enumerate(schedules[str(seed)], 1):
                model.train()
                totals, zeros = dict.fromkeys(('loss', 'mae', 'constraint'), 0.0), 0
                for indices in selections:
                    batch = fixed_batch([datasets['dev'].flights[i] for i in indices.tolist()], max_length, device)
                    optimizer.zero_grad(set_to_none=True)
                    z, prediction = model(batch['x'], batch['w'])
                    mae = masked_mae(batch['x'], prediction, batch['mask'])
                    penalty, zero_groups = constraint_loss(z, batch['unit'], batch['cycle'], variant)
                    loss = mae + penalty
                    if not torch.isfinite(loss):
                        raise RuntimeError(f'Nonfinite loss: {variant}/{seed}/{epoch}')
                    loss.backward()
                    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise RuntimeError(f'Nonfinite/missing gradients: {variant}/{seed}/{epoch}')
                    optimizer.step()
                    for key, value in (('loss', loss), ('mae', mae), ('constraint', penalty)):
                        totals[key] += value.item()
                    zeros += zero_groups
                logs.append({'epoch': epoch, 'batches': batches, 'sampled_flights': batches * 20,
                             **{key: value / batches for key, value in totals.items()}, 'zero_z_unit_batches': zeros})
                pd.DataFrame(logs).to_csv(folder / 'losses.csv', index=False, float_format='%.17g')
                print(f'{variant} seed={seed} epoch={epoch}/{epochs}: loss={logs[-1]["loss"]:.6f} MAE={logs[-1]["mae"]:.6f} constraint={logs[-1]["constraint"]:.6f}', flush=True)
            model.eval()
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            assert any(not torch.equal(state[key], initial[key]) for key in state)
            values = {split: predict_z(model, dataset, device) for split, dataset in datasets.items()}
            scaling = {'min': values['dev'].double().min().item(), 'max': values['dev'].double().max().item(),
                       'count': len(values['dev']), 'fit_split': 'dev'}
            torch.save({'state_dict': state, 'initial_state_dict': initial, 'initial_sha256': initial_hash,
                        'variant': variant, 'seed': seed, 'epoch': epochs, 'settings': settings, 'normalization': processed['normalization'],
                        'columns': processed['columns'], 'hi_minmax': scaling, 'z': values}, folder / 'checkpoint.pt')
            del model, optimizer
    # Labels are loaded only after all variant/seed trainings finish.
    references = {}
    for split in ('dev', 'test'):
        hi = read_reference(config['data'][f'hi_{split}_path'], processed[split])
        rul = read_rul(config['data']['raw_path'], split, processed[split])
        references[split] = hi.merge(rul, on=['unit', 'cycle'], validate='one_to_one').set_index(['unit', 'cycle'])
    rows, unit_rows = [], []
    for variant in VARIANTS:
        for seed in seeds:
            folder = out / variant / f'seed_{seed}'
            saved = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
            model = ReferenceAE(max_length).to(device).eval()
            model.load_state_dict(saved['state_dict'])
            report, tables = {}, []
            for split, dataset in datasets.items():
                z = predict_z(model, dataset, device)
                torch.testing.assert_close(z, saved['z'][split], rtol=0, atol=0)
                if split == 'dev':
                    assert saved['hi_minmax']['min'] == z.double().min().item() and saved['hi_minmax']['max'] == z.double().max().item()
                scale = saved['hi_minmax']
                hi = (z.double().numpy() - scale['min']) / (scale['max'] - scale['min']) if scale['max'] != scale['min'] else np.full(len(z), np.nan)
                keys = pd.MultiIndex.from_tuples([tuple(key) for key in settings['keys'][split]], names=['unit', 'cycle'])
                table = references[split].loc[keys].reset_index()
                table['split'], table['z'], table['hi_scaled'] = split, z.numpy(), hi
                report[split] = metric_table(table)
                rank_rows = ranking(table)
                for row in rank_rows:
                    unit_rows.append({'variant': variant, 'seed': seed, 'split': split, **row})
                overall = rank_rows[-1]
                metric_values = report[split].get('overall', {})
                rows.append({'variant': variant, 'seed': seed, 'split': split,
                             **{key: value for key, value in overall.items() if key != 'unit'},
                             **{key: metric_values.get(key, {}).get('value') for key in ('mae', 'mape', 'monotonicity', 'trend', 'failure_consistency', 'mi')},
                             'direction': 'rising' if overall['spearman_z_cycle'] > 0 else ('falling' if overall['spearman_z_cycle'] < 0 else 'constant_or_mixed'),
                             'dev_min': scale['min'], 'dev_max': scale['max'], 'mode': settings['mode']})
                tables.append(table)
            exported = pd.concat(tables, ignore_index=True)
            exported.to_csv(folder / 'predictions.csv', index=False, float_format='%.17g')
            dump_json(folder / 'metrics.json', report)
            if seed == 42:
                disk = pd.read_csv(folder / 'predictions.csv', float_precision='round_trip')
                for split in ('dev', 'test'):
                    plot_split(disk[disk.split == split], report[split], folder / f'hi_curves_{split}.png', variant, epochs)
            del model
    results = pd.DataFrame(rows)
    results.to_csv(out / 'results_per_seed.csv', index=False, float_format='%.17g')
    pd.DataFrame(unit_rows).to_csv(out / 'ranking_per_unit.csv', index=False, float_format='%.17g')
    metric_names = ('accuracy', 'order_hi', 'order_z', 'spearman_hi_cycle', 'spearman_z_cycle', 'mae', 'mape', 'monotonicity', 'trend', 'failure_consistency', 'mi')
    summary = []
    for (variant, split), group in results.groupby(['variant', 'split'], sort=False):
        row = {'variant': variant, 'split': split, 'seeds': len(group), 'mode': settings['mode']}
        for metric in metric_names:
            finite = group[metric].dropna()
            row[f'{metric}_valid_seeds'] = len(finite)
            row[f'{metric}_mean'] = float(finite.mean()) if len(finite) else None
            row[f'{metric}_std'] = float(finite.std(ddof=0)) if len(finite) else None
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(out / 'results_summary.csv', index=False, float_format='%.17g')
    comparison = summary[['variant', 'split', 'seeds', 'accuracy_mean', 'accuracy_std', 'spearman_hi_cycle_mean']].copy()
    comparison['source'] = 'this run, 20-epoch main setting' if epochs == 20 and lr == 1e-4 and not smoke else f'this run, {settings["mode"]}, epochs={epochs}, lr={lr}'
    prior = []
    for name, dev, test, count, source in [('OCALD_formal', .5435, .5235, 1, 'user-provided formal OCALD; historical single run'),
                                         ('OCALD_standardized_mu_I', .5826, .5599, 1, 'README D1/D2; historical seed 42'),
                                         ('F1_statistics', .9349, .9409, 5, 'README A/B F1; historical 5-seed mean')]:
        for split, accuracy in [('dev', dev), ('test', test)]:
            prior.append({'variant': name, 'split': split, 'seeds': count, 'accuracy_mean': accuracy, 'source': source})
    pd.concat([comparison, pd.DataFrame(prior)], ignore_index=True).to_csv(out / 'comparison.csv', index=False, float_format='%.17g')
    if protected_files(config) != hashes:
        raise RuntimeError('An existing source/config/data file changed')
    verify_run(out, device, write_result=True)
    notes = ['Paper-style diagnostic baseline, not an exact reproduction of the reference notebook.',
             f'Mode={settings["mode"]}; T_max={max_length}; Adam lr={lr}; epochs={epochs}; seeds={seeds}.',
             'Training reads no HI/RUL; labels loaded after all model training. Final weights only, no selection.',
             'MAE is mean of flight means over valid elements. Cycle/unit only form constraints/evaluation.',
             'NG uses differences, not cycle-normalized derivatives. Pearson is per-unit. No latent L2 normalization.',
             'All Conv kernels=11. Zero padding is excluded from MAE but may convey sequence length through Conv/Flatten.',
             'HI is dev Min-Max of Z, without direction flipping or test clipping. Positive signed Spearman is flagged rising.',
             'Order on scaled HI is softplus(HI_late-HI_early); also report Z order because scaling changes magnitudes.',
             'Historical comparison rows have different seed counts, model capacities and training budgets.',
             'Results can diagnose this data/model combination, but cannot uniquely identify a cause of OCALD failure.',
             'MAE/MAPE average flights; ranking/order/signed Spearman average units; seed std uses ddof=0.',
             'Unavailable metrics retain reasons in metrics.json and valid-seed counts in summary CSV.', '',
             summary[['variant', 'split', 'accuracy_mean', 'accuracy_std', 'spearman_hi_cycle_mean', 'order_hi_mean']].to_string(index=False), '',
             'Direction flags:', results[['variant', 'seed', 'split', 'direction']].to_string(index=False)]
    (out / 'summary.txt').write_text('\n'.join(notes) + '\n', encoding='utf-8')
    print(summary[['variant', 'split', 'accuracy_mean', 'accuracy_std', 'spearman_hi_cycle_mean', 'order_hi_mean']].to_string(index=False), flush=True)
    print(f'PASS: baseline_ref {settings["mode"]}; output={out}', flush=True)


def verify_run(out, device, write_result=False):
    out = Path(out)
    settings = json.loads((out / 'run_settings.json').read_text())
    config = settings['config']
    if protected_files(config) != settings['protected_sha256']:
        raise ValueError('Protected file hashes differ from run')
    processed = load_processed(config)
    datasets = select_datasets(processed, settings['mode'] == 'smoke')
    schedule = torch.load(out / 'batch_schedule.pt', map_location='cpu', weights_only=True)
    unit_results = pd.read_csv(out / 'ranking_per_unit.csv', float_precision='round_trip')
    seed_results = pd.read_csv(out / 'results_per_seed.csv', float_precision='round_trip')
    seed_summary = pd.read_csv(out / 'results_summary.csv', float_precision='round_trip')
    initial_hashes = {}
    for seed in settings['seeds']:
        repeat = make_schedule(datasets['dev'].flights, settings['epochs'], settings['batches_per_epoch'], seed)
        assert torch.equal(repeat, schedule['schedules'][str(seed)])
        for variant in VARIANTS:
            folder = out / variant / f'seed_{seed}'
            saved = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
            assert saved['variant'] == variant and saved['seed'] == seed and saved['epoch'] == settings['epochs']
            initial_hash = hashlib.sha256(b''.join(v.numpy().tobytes() for v in saved['initial_state_dict'].values())).hexdigest()
            assert initial_hash == saved['initial_sha256']
            if seed in initial_hashes:
                assert initial_hashes[seed] == initial_hash
            initial_hashes[seed] = initial_hash
            assert any(not torch.equal(saved['state_dict'][k], v) for k, v in saved['initial_state_dict'].items())
            logs = pd.read_csv(folder / 'losses.csv', float_precision='round_trip')
            assert logs.epoch.tolist() == list(range(1, settings['epochs'] + 1))
            assert (logs.batches == settings['batches_per_epoch']).all()
            np.testing.assert_allclose(logs.loss, logs.mae + logs.constraint, rtol=1e-6, atol=1e-7)
            if variant == 'V0':
                assert (logs.constraint == 0).all()
            model = ReferenceAE(settings['T_max']).to(device).eval()
            model.load_state_dict(saved['state_dict'])
            table = pd.read_csv(folder / 'predictions.csv', float_precision='round_trip')
            metrics = json.loads((folder / 'metrics.json').read_text())
            for split, dataset in datasets.items():
                subset = table[table.split == split].reset_index(drop=True)
                assert subset[['unit', 'cycle']].values.tolist() == settings['keys'][split]
                fresh = predict_z(model, dataset, device)
                np.testing.assert_array_equal(fresh.numpy(), subset.z)
                np.testing.assert_array_equal(fresh.numpy(), saved['z'][split].numpy())
                scale = saved['hi_minmax']
                if split == 'dev':
                    assert scale['min'] == fresh.double().min().item() and scale['max'] == fresh.double().max().item()
                if scale['min'] != scale['max']:
                    expected = (fresh.double().numpy() - scale['min']) / (scale['max'] - scale['min'])
                    np.testing.assert_array_equal(expected, subset.hi_scaled)
                else:
                    assert subset.hi_scaled.isna().all()
                fresh_rank = pd.DataFrame(ranking(subset))
                stored_rank = unit_results[(unit_results.variant == variant) & (unit_results.seed == seed)
                                           & (unit_results.split == split)].reset_index(drop=True)
                assert fresh_rank.unit.tolist() == stored_rank.unit.tolist()
                for field in fresh_rank.columns.drop('unit'):
                    np.testing.assert_allclose(fresh_rank[field].to_numpy(dtype=float), stored_rank[field],
                                               rtol=0, atol=1e-14, equal_nan=True)
                stored_seed = seed_results[(seed_results.variant == variant) & (seed_results.seed == seed)
                                           & (seed_results.split == split)]
                assert len(stored_seed) == 1
                for field, value in fresh_rank.iloc[-1].items():
                    if field != 'unit':
                        np.testing.assert_allclose(float(value) if value is not None else np.nan,
                                                   stored_seed[field].iloc[0], rtol=0, atol=1e-14, equal_nan=True)
                assert metric_table(subset) == metrics[split]
            del model
    for _, row in seed_summary.iterrows():
        group = seed_results[(seed_results.variant == row.variant) & (seed_results.split == row.split)]
        assert len(group) == row.seeds == len(settings['seeds'])
        for mean_column in seed_summary.columns:
            if mean_column.endswith('_mean'):
                field = mean_column[:-5]
                valid = group[field].dropna()
                assert row[field + '_valid_seeds'] == len(valid)
                expected_mean = valid.mean() if len(valid) else np.nan
                expected_std = valid.std(ddof=0) if len(valid) else np.nan
                np.testing.assert_allclose(row[mean_column], expected_mean, rtol=0, atol=1e-12, equal_nan=True)
                np.testing.assert_allclose(row[field + '_std'], expected_std, rtol=0, atol=1e-12, equal_nan=True)
    if write_result:
        dump_json(out / 'verification.json', {'shape_mask_pairs_constraint_gradient_HI_checks': True,
            'shared_initialization_per_seed': True, 'schedule_regeneration_exact': True,
            'checkpoint_reload_predictions_exact': True, 'saved_metrics_recomputed': True,
            'saved_ranking_and_seed_summary_recomputed': True, 'protected_files_unchanged': True,
            'label_training_path': 'HI/RUL loaded only after all model trainings', 'device': str(device), 'mode': settings['mode']})
    print(f'PASS: saved baseline checkpoints/predictions/metrics/schedules; output={out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--run-dir')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.epochs <= 0 or not math.isfinite(args.lr) or args.lr <= 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Need positive epochs/lr and distinct seeds')
    if args.verify_only:
        if args.run_dir is None or args.smoke:
            parser.error('--verify-only requires --run-dir and excludes --smoke')
        saved_settings = json.loads((Path(args.run_dir) / 'run_settings.json').read_text())
        if saved_settings['config'] != config:
            parser.error('Run configuration differs')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            torch.set_num_threads(1)
        verify_run(args.run_dir, device)
    else:
        if args.run_dir is not None:
            parser.error('--run-dir is only for --verify-only; output directories are always new')
        run(config, args.epochs, args.lr, args.seeds, args.smoke)


if __name__ == '__main__':
    main()
