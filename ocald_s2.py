"""S2 diagnostic: original OCALD ELBO plus encoder-backpropagated ranking."""

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
from torch.nn import functional as F

from baseline_ref import finite_report, protected_files
from data import FlightDataset, collate_fn, load_config, load_processed, read_reference, read_rul
from diagnose import apply_transform, extract_blocks, fit_transform, pair_schedule
from diagnostics_features import ranking_rows
from losses import head_l2, hi_loss, order_loss, reconstruction_loss, representation_loss
from metrics import evaluate_metrics
from model import HIHead, OCALD, time_positions
from train import freeze_representation, sample_pairs, scale_hi, valid_pair_pools, validate_hi_config


LAMBDAS = (0.1, 1.0, 10.0)
SEEDS = (42, 43, 44)
E1_RESULTS = 'outputs/baseline_ref/20261012T080807303596Z/results_summary.csv'


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_hash(state):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in state.values())).hexdigest()


def protected_hashes(config):
    result = protected_files(config)
    for path in ('diagnose.py', 'diagnostics_features.py', 'diagnostics_latent.py', 'baseline_ref.py', 'README_baseline_ref.md'):
        result[path] = file_hash(path)
    return result


def datasets_for(processed, smoke):
    datasets = {}
    for split in ('dev', 'test'):
        dataset = FlightDataset(processed, split)
        expected = list(range(1, 10)) if split == 'dev' else list(range(10, 16))
        if sorted({f['unit'] for f in dataset.flights}) != expected:
            raise ValueError(f'Unexpected {split} unit set')
        keys = [(f['unit'], f['cycle']) for f in dataset.flights]
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate flight keys')
        if smoke:
            selected = []
            for unit in expected:
                flights = sorted((f for f in dataset.flights if f['unit'] == unit), key=lambda f: f['cycle'])
                indices = np.unique(np.linspace(0, len(flights) - 1, min(8, len(flights))).round().astype(int))
                selected.extend(flights[i] for i in indices)
            dataset.flights = selected
        datasets[split] = dataset
    return datasets


def move_batch(flights, device):
    return {key: value.to(device) for key, value in collate_fn(flights).items()}


def joint_schedule(flights, epochs, batches, seed):
    units = torch.tensor([f['unit'] for f in flights])
    cycles = torch.tensor([f['cycle'] for f in flights])
    groups = [torch.where(units == unit)[0] for unit in torch.unique(units)]
    if len(groups) < 4 or any(len(g) < 4 or cycles[g].max() - cycles[g].min() < 5 for g in groups):
        raise ValueError('Grouped sampling needs >=4 units with >=4 flights and valid cycle gaps')
    flight_rng = torch.Generator().manual_seed(seed)
    batch_list, pair_list = [], []
    # Existing sampler uses global CPU RNG; isolate it from initialization/noise.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        for _ in range(epochs * batches):
            selections = []
            for group_index in torch.randperm(len(groups), generator=flight_rng)[:4]:
                group = groups[int(group_index)]
                while True:
                    selected = group[torch.randperm(len(group), generator=flight_rng)[:4]]
                    if cycles[selected].max() - cycles[selected].min() >= 5:
                        selections.append(selected)
                        break
            indices = torch.cat(selections)
            u, c = units[indices], cycles[indices]
            assert len(torch.unique(indices)) == 16
            assert torch.unique(u, return_counts=True)[1].tolist() == [4] * 4
            pairs = sample_pairs(valid_pair_pools(u, c, 5), 16)
            assert (u[pairs[:, 0]] == u[pairs[:, 1]]).all()
            assert (c[pairs[:, 1]] - c[pairs[:, 0]] >= 5).all()
            batch_list.append(indices)
            pair_list.append(pairs)
    return {'flights': torch.stack(batch_list).reshape(epochs, batches, 16),
            'pairs': torch.stack(pair_list).reshape(epochs, batches, 16, 2)}


def encoder_order_norm(model, order):
    params = list(model.sensor_encoder.parameters())
    grads = torch.autograd.grad(order, params, retain_graph=True, allow_unused=True)
    # Variance/C output heads are not on the ranking path; GRU and mu_I are.
    names = list(model.sensor_encoder.named_parameters())
    for (name, _), grad in zip(names, grads):
        if name.startswith(('gru.', 'mu_i.')) and grad is None:
            raise RuntimeError(f'Missing ranking gradient: {name}')
    valid = [g for g in grads if g is not None]
    if not valid or not all(torch.isfinite(g).all() for g in valid):
        raise RuntimeError('Invalid encoder ranking gradients')
    return torch.stack([g.detach().square().sum() for g in valid]).sum().sqrt().item()


def check_mechanisms(config, batch, pairs):
    model = OCALD(config).to(batch['x'].device)
    model.train()  # cuDNN GRU backward requires a training-mode forward.
    original_mask = batch['mask'].clone()
    output = model(batch['x'], batch['w'], batch['mask'])
    r = model.hi_head(output['mu_i'])
    order = order_loss(r[pairs[:, 0]], r[pairs[:, 1]], 1)
    norm = encoder_order_norm(model, order)
    if norm <= 0:
        raise RuntimeError('Ranking gradient did not reach encoder')
    other = list(model.decoder.parameters()) + list(model.conditional_prior.parameters())
    assert all(g is None for g in torch.autograd.grad(order, other, allow_unused=True, retain_graph=True))
    losses = representation_loss(batch['x'], batch['mask'], output, 1)
    torch.testing.assert_close(losses['loss'], losses['reconstruction'] + losses['kl_i'] + losses['kl_c'])
    before = reconstruction_loss(batch['x'], output['x_hat'], batch['mask'], 1)
    changed_x = batch['x'].masked_fill(~batch['mask'][:, :, None], float('nan'))
    changed_prediction = output['x_hat'].masked_fill(~batch['mask'][:, :, None], float('nan'))
    torch.testing.assert_close(before, reconstruction_loss(changed_x, changed_prediction, batch['mask'], 1), rtol=0, atol=0)
    model.eval()
    with torch.no_grad():
        posterior = model.encode(batch['x'], batch['mask'])
        for a, b in zip(posterior, model.encode(changed_x, batch['mask'])):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        extended_mask = F.pad(batch['mask'], (0, 8), value=False)
        extended_x = F.pad(batch['x'], (0, 0, 0, 8), value=123)
        for a, b in zip(posterior, model.encode(extended_x, extended_mask)):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
        mu_i, _, mu_c, _ = posterior
        baseline = model.decode(mu_i, mu_c, time_positions(batch['mask']))
        extended = model.decode(mu_i, mu_c, time_positions(extended_mask))
        torch.testing.assert_close(baseline[batch['mask']], extended[extended_mask], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(model.predict_hi(batch['x'], batch['mask']), model.predict_hi(extended_x, extended_mask), rtol=1e-6, atol=1e-6)
    assert torch.equal(batch['mask'], original_mask)
    assert (batch['unit'][pairs[:, 0]] == batch['unit'][pairs[:, 1]]).all()
    assert (batch['cycle'][pairs[:, 1]] - batch['cycle'][pairs[:, 0]] >= 5).all()
    assert order_loss(torch.tensor([0.]), torch.tensor([1.]), 1) < order_loss(torch.tensor([1.]), torch.tensor([0.]), 1)
    print(f'PASS: S2 direction/mask/ELBO/path checks; encoder ranking gradient norm={norm:.9g}', flush=True)
    return {'initial_encoder_order_gradient_norm': norm, 'mask_padding_checked': True,
            'order_only_decoder_prior_gradients_absent': True, 'pair_direction_checked': True}


def folder_for(root, weight, seed):
    return Path(root) / f'lambda_{weight:g}' / f'seed_{seed}'


def load_s2(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['stage'] != 'ocald_s2':
        raise ValueError('Expected an S2 checkpoint, not a formal two-stage checkpoint')
    validate_hi_config(saved['config'])
    model = OCALD(saved['config']).to(device)
    model.load_state_dict(saved['state_dict'], strict=True)
    return model.eval(), saved


def train_joint(config, dataset, datasets, schedules, weight, seed, epochs, folder, processed, device):
    torch.manual_seed(seed)
    model = OCALD(config).to(device)
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0)
    logs = []
    for epoch in range(epochs):
        model.train()
        totals = dict.fromkeys(('loss', 'elbo', 'reconstruction', 'kl_i', 'kl_c', 'order', 'l2'), 0.)
        order_norm, clip_norms = None, []
        for step, (indices, cpu_pairs) in enumerate(zip(schedules['flights'][epoch], schedules['pairs'][epoch])):
            batch = move_batch([dataset.flights[i] for i in indices.tolist()], device)
            pairs = cpu_pairs.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch['x'], batch['w'], batch['mask'])
            elbo = representation_loss(batch['x'], batch['mask'], output, 1.)
            scores = model.hi_head(output['mu_i'])
            order = order_loss(scores[pairs[:, 0]], scores[pairs[:, 1]], 1.)
            l2 = head_l2(model.hi_head)
            loss = elbo['loss'] + weight * order + 1e-4 * l2
            if not all(torch.isfinite(v) for v in (*elbo.values(), order, l2, loss)):
                raise RuntimeError(f'Nonfinite S2 loss: lambda={weight}, seed={seed}, epoch={epoch+1}')
            if step == 0:
                order_norm = encoder_order_norm(model, order)
                if epoch == 0 and order_norm <= 0:
                    raise RuntimeError('First real training batch has zero encoder ranking gradient')
            loss.backward()
            clip_norms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True).item())
            optimizer.step()
            values = {'loss': loss, 'elbo': elbo['loss'], 'reconstruction': elbo['reconstruction'],
                      'kl_i': elbo['kl_i'], 'kl_c': elbo['kl_c'], 'order': order, 'l2': l2}
            for key, value in values.items():
                totals[key] += value.detach().item()
        batches = len(schedules['flights'][epoch])
        logs.append({'epoch': epoch + 1, 'batches': batches, 'sampled_flights': batches * 16, 'sampled_pairs': batches * 16,
                     **{key: value / batches for key, value in totals.items()}, 'encoder_order_grad_norm': order_norm,
                     'mean_preclip_grad_norm': float(np.mean(clip_norms)), 'lambda': weight, 'seed': seed})
        pd.DataFrame(logs).to_csv(folder / 'losses.csv', index=False, float_format='%.17g')
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(f'S2 lambda={weight:g} seed={seed} epoch={epoch+1}/{epochs}: ' + ' '.join(f'{key}={logs[-1][key]:.6f}' for key in ('loss', 'reconstruction', 'kl_i', 'kl_c', 'order')), flush=True)
    rng = torch.cuda.get_rng_state(device) if device.type == 'cuda' else torch.get_rng_state()
    rng_digest = hashlib.sha256(rng.numpy().tobytes()).hexdigest()
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    for module in ('sensor_encoder', 'conditional_prior', 'decoder', 'hi_head'):
        assert any(not torch.equal(value, initial[key]) for key, value in state.items() if key.startswith(module + '.'))
    model.eval()
    blocks = {split: extract_blocks(model, data, 16, device) for split, data in datasets.items()}
    with torch.no_grad():
        device_scores = {split: model.hi_head(values['I'][0]) for split, values in blocks.items()}
        scores = {split: r.cpu() for split, r in device_scores.items()}
        raw = {split: torch.sigmoid(-r).cpu() for split, r in device_scores.items()}
    scaling = {'min': raw['dev'].min().item(), 'max': raw['dev'].max().item(), 'count': len(raw['dev']), 'fit_split': 'dev'}
    saved = {'stage': 'ocald_s2', 'mode': 'smoke' if epochs == 3 else 'full', 'epoch': epochs, 'lambda': weight, 'seed': seed,
             'config': config, 'state_dict': state, 'initial_state_dict': initial, 'initial_sha256': state_hash(initial),
             'normalization': processed['normalization'], 'columns': processed['columns'], 'hi_minmax': scaling,
             'posterior_means': {split: {name: values[0].cpu() for name, values in data.items()} for split, data in blocks.items()},
             'scores': scores, 'raw_hi': raw, 'reparameterization_rng_digest': rng_digest,
             'first_encoder_order_grad_norm': logs[0]['encoder_order_grad_norm']}
    torch.save(saved, folder / 'checkpoint.pt')


def ablate(model, means, datasets, weight, seed, device):
    rows = []
    with torch.no_grad():
        for split, dataset in datasets.items():
            mu_i, mu_c = means[split]['I'].to(device), means[split]['C'].to(device)
            settings = {'posterior_mean': (mu_i, mu_c), 'zero_both': (torch.zeros_like(mu_i), torch.zeros_like(mu_c)),
                        'zero_i': (torch.zeros_like(mu_i), mu_c), 'zero_c': (mu_i, torch.zeros_like(mu_c))}
            totals = dict.fromkeys(settings, 0.)
            elements = 0
            for start in range(0, len(dataset), 16):
                batch = move_batch(dataset.flights[start:start+16], device)
                count = len(batch['x'])
                elements += int(batch['mask'].sum()) * 14
                for name, (zi, zc) in settings.items():
                    prediction = model.decode(zi[start:start+count], zc[start:start+count], time_positions(batch['mask']))
                    losses = reconstruction_loss(batch['x'], prediction, batch['mask'], 1)
                    assert torch.isfinite(losses).all()
                    sse = (batch['x'][batch['mask']].double() - prediction[batch['mask']].double()).square().sum()
                    torch.testing.assert_close(losses.double().sum() * 2, sse, rtol=1e-6, atol=1e-6)
                    totals[name] += losses.double().sum().item()
            for name, total in totals.items():
                rows.append({'lambda': weight, 'seed': seed, 'split': split, 'setting': name,
                             'reconstruction_mean': total / len(dataset), 'element_mse': total * 2 / elements,
                             'flights': len(dataset), 'valid_elements': elements})
    return rows


def train_probes(model, saved, datasets, schedule, folder, device, epochs):
    model.zero_grad(set_to_none=True)
    freeze_representation(model)
    for parameter in model.hi_head.parameters():
        parameter.requires_grad_(False)
    before = state_hash(model.state_dict())
    blocks = {split: extract_blocks(model, dataset, 16, device) for split, dataset in datasets.items()}
    for split in blocks:
        for name in ('I', 'C'):
            torch.testing.assert_close(blocks[split][name][0].cpu(), saved['posterior_means'][split][name], rtol=0, atol=0)
    raw = {name: {split: values[name][0] for split, values in blocks.items()} for name in ('I', 'C')}
    raw['IC'] = {split: torch.cat((raw['I'][split], raw['C'][split]), 1) for split in blocks}
    transforms = {name: fit_transform(values['dev']) for name, values in raw.items()}
    for key in ('mean', 'std', 'denominator'):
        concatenated = torch.cat((transforms['I'][key], transforms['C'][key]))
        torch.testing.assert_close(transforms['IC'][key], concatenated, rtol=1e-12, atol=1e-12)
        transforms['IC'][key] = concatenated
    inputs = {name: {split: apply_transform(values, transforms[name]) for split, values in groups.items()} for name, groups in raw.items()}
    all_rows = []
    seed = saved['seed']
    units, cycles = blocks['dev']['I'][1:]
    for name in ('I', 'C', 'IC'):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            head = HIHead(inputs[name]['dev'].shape[1], 16).to(device)
        initial = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        optimizer = torch.optim.Adam(head.parameters(), lr=.001, weight_decay=0)
        logs = []
        for epoch, cpu_pairs in enumerate(schedule, 1):
            pairs = cpu_pairs.to(device)
            totals = dict.fromkeys(('loss', 'order', 'l2'), 0.)
            for start in range(0, len(pairs), 128):
                pair = pairs[start:start+128]
                optimizer.zero_grad(set_to_none=True)
                loss = hi_loss(head(inputs[name]['dev'][pair[:, 0]]), head(inputs[name]['dev'][pair[:, 1]]), head, 1, 1e-4)
                if not all(torch.isfinite(v) for v in loss.values()):
                    raise RuntimeError('Nonfinite S2 probe loss')
                loss['loss'].backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
                optimizer.step()
                for key in totals:
                    totals[key] += loss[key].item() * len(pair)
            logs.append({'epoch': epoch, 'pairs': len(pairs), **{key: value/len(pairs) for key, value in totals.items()}})
        head.eval()
        probe_folder = folder / 'probes' / name
        probe_folder.mkdir(parents=True)
        pd.DataFrame(logs).to_csv(probe_folder / 'losses.csv', index=False, float_format='%.17g')
        state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        assert any(not torch.equal(state[key], initial[key]) for key in initial)
        with torch.no_grad():
            scores = {split: head(values).cpu() for split, values in inputs[name].items()}
            hi = {split: torch.sigmoid(-head(values)).cpu() for split, values in inputs[name].items()}
        torch.save({'state_dict': state, 'initial_state_dict': initial, 'transform': transforms[name],
                    'input_dim': inputs[name]['dev'].shape[1], 'seed': seed, 'epoch': epochs,
                    'source_sha256': file_hash(folder / 'checkpoint.pt'), 'scores': scores, 'raw_hi': hi}, probe_folder / 'head.pt')
        predictions = []
        for split, dataset in datasets.items():
            split_units, split_cycles = blocks[split]['I'][1:]
            rows = ranking_rows(name, split, seed, scores[split].to(device), split_units, split_cycles, 5, 1)
            for row in rows:
                row['lambda'], row['probe'] = saved['lambda'], name
                del row['feature_set']
                all_rows.append(row)
            predictions.extend({'split': split, 'unit': f['unit'], 'cycle': f['cycle'], 'r': r.item(), 'hi_raw': h.item()}
                               for f, r, h in zip(dataset.flights, scores[split], hi[split]))
        pd.DataFrame(predictions).to_csv(probe_folder / 'predictions.csv', index=False, float_format='%.17g')
        print(f'probe lambda={saved["lambda"]:g} seed={seed} block={name}: {epochs} epochs done', flush=True)
    assert state_hash(model.state_dict()) == before
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    i = torch.load(folder / 'probes/I/head.pt', weights_only=True, map_location='cpu')['initial_state_dict']
    c = torch.load(folder / 'probes/C/head.pt', weights_only=True, map_location='cpu')['initial_state_dict']
    assert all(torch.equal(i[key], c[key]) for key in i)
    return all_rows


def metric_entry(table, column):
    if table[column].isna().all():
        return {'available': False, 'reason': 'Dev raw HI range is zero'}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        report = evaluate_metrics(table[column], table.hi_gt, table.unit, table.cycle, table.rul, 42)
    return {'available': True, **finite_report(report), 'warnings': sorted({str(w.message) for w in caught})}


def plot_curves(table, reports, path, weight, epochs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    groups = list(table.groupby('unit', sort=True))
    figure, axes = plt.subplots(len(groups), 2, figsize=(14, 2.5*len(groups)), layout='constrained', squeeze=False)
    figure.suptitle(f'OCALD S2 | lambda={weight:g} | seed=42 | {epochs} joint epochs | {table.split.iloc[0]}')
    for row, (unit, group) in enumerate(groups):
        group = group.sort_values('cycle')
        for column, (scale, field) in enumerate((('raw', 'hi_raw'), ('scaled', 'hi_scaled'))):
            axis = axes[row, column]
            axis.plot(group.cycle, group.hi_gt, color='black', linewidth=1, label='Reference HI')
            report = reports[scale]
            details = 'Scaled unavailable: zero dev range'
            if report['available']:
                line, = axis.plot(group.cycle, group[field], linewidth=1.1, label=f'Predicted HI ({scale})')
                assert np.array_equal(line.get_xdata(), group.cycle) and np.array_equal(line.get_ydata(), group[field])
                metrics = report['per_unit'][str(unit)]
                details = ' '.join(f"{label}={metrics[key]['value']:.3f}" if metrics[key]['value'] is not None else f'{label}=N/A'
                                   for label, key in [('MAE', 'mae'), ('MAPE%', 'mape'), ('M', 'monotonicity'), ('rho', 'spearman'), ('MI', 'mi')])
            axis.set_title(f'unit {unit} | {scale}\n{details}', fontsize=9)
            axis.set_xlabel('Flight cycle')
            axis.set_ylabel('HI')
            axis.grid(alpha=.25)
            axis.legend(fontsize=8)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_training(log_path, output, weight):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    logs = pd.read_csv(log_path, float_precision='round_trip')
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    figure.suptitle(f'OCALD S2 | lambda={weight:g} | seed=42 | actual training terms')
    for axis, key in zip(axes.flat, ('reconstruction', 'kl_i', 'kl_c', 'order')):
        line, = axis.plot(logs.epoch, logs[key])
        assert np.array_equal(line.get_ydata(), logs[key])
        axis.set_title(key)
        axis.set_xlabel('Epoch')
        axis.grid(alpha=.25)
    figure.savefig(output, dpi=140)
    plt.close(figure)


def summarize(frame, groups, metrics):
    rows = []
    for key, data in frame.groupby(groups, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, key))
        row['seeds'] = data.seed.nunique()
        for metric in metrics:
            values = data[metric].dropna()
            row[metric+'_valid_seeds'] = len(values)
            row[metric+'_mean'] = values.mean() if len(values) else None
            row[metric+'_std'] = values.std(ddof=0) if len(values) else None
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_run(root, settings, processed, datasets, device):
    # This is the first label read, after ALL joint and probe training completes.
    references = {}
    config = settings['config']
    for split in ('dev', 'test'):
        hi = read_reference(config['data'][f'hi_{split}_path'], processed[split])
        rul = read_rul(config['data']['raw_path'], split, processed[split])
        references[split] = hi.merge(rul, on=['unit', 'cycle'], validate='one_to_one').set_index(['unit', 'cycle'])
    results, ranking_all = [], []
    for weight in LAMBDAS:
        for seed in SEEDS:
            folder = folder_for(root, weight, seed)
            model, saved = load_s2(folder / 'checkpoint.pt', device)
            tables, reports = [], {}
            for split, dataset in datasets.items():
                keys = pd.MultiIndex.from_tuples([(f['unit'], f['cycle']) for f in dataset.flights], names=['unit', 'cycle'])
                table = references[split].loc[keys].reset_index()
                raw = saved['raw_hi'][split]
                scaling = saved['hi_minmax']
                scaled = scale_hi(raw, scaling).numpy() if scaling['min'] != scaling['max'] else np.full(len(raw), np.nan)
                # CSV is read as float64. Use the same exact float32 values in
                # float64 here, so FC/MI reductions match saved-table rechecks.
                table['split'], table['r'], table['hi_raw'], table['hi_scaled'] = (
                    split, saved['scores'][split].numpy().astype(np.float64),
                    raw.numpy().astype(np.float64), scaled.astype(np.float64))
                reports[split] = {scale: metric_entry(table, column) for scale, column in (('raw', 'hi_raw'), ('scaled', 'hi_scaled'))}
                units = torch.tensor(table.unit.to_numpy(), device=device)
                cycles = torch.tensor(table.cycle.to_numpy(), device=device)
                rows = ranking_rows('joint', split, seed, saved['scores'][split].to(device), units, cycles, 5, 1)
                for row in rows:
                    row['lambda'] = weight
                    del row['feature_set']
                    ranking_all.append(row)
                overall = rows[-1]
                row = {'lambda': weight, 'seed': seed, 'split': split, **{key: value for key, value in overall.items() if key != 'unit'},
                       'hi_eq_zero': int((raw == 0).sum()), 'hi_eq_one': int((raw == 1).sum())}
                for scale in ('raw', 'scaled'):
                    metric_values = reports[split][scale].get('overall', {})
                    for metric in ('mae', 'mape', 'monotonicity', 'trend', 'failure_consistency', 'mi'):
                        row[f'{scale}_{metric}'] = metric_values.get(metric, {}).get('value')
                results.append(row)
                tables.append(table)
            combined = pd.concat(tables, ignore_index=True)
            combined.to_csv(folder / 'predictions.csv', index=False, float_format='%.17g')
            dump_json(folder / 'metrics.json', reports)
            if seed == 42:
                disk = pd.read_csv(folder / 'predictions.csv', float_precision='round_trip')
                for split in ('dev', 'test'):
                    plot_curves(disk[disk.split == split], reports[split], folder / f'hi_curves_{split}.png', weight, settings['epochs'])
                plot_training(folder / 'losses.csv', folder / 'training_curves.png', weight)
    frame = pd.DataFrame(results)
    frame.to_csv(root / 'results_per_seed.csv', index=False, float_format='%.17g')
    pd.DataFrame(ranking_all).to_csv(root / 'ranking_per_unit.csv', index=False, float_format='%.17g')
    metrics = ['accuracy', 'order', 'spearman_hi_cycle'] + [f'{scale}_{metric}' for scale in ('raw', 'scaled')
        for metric in ('mae', 'mape', 'monotonicity', 'trend', 'failure_consistency', 'mi')]
    summary = summarize(frame, ['lambda', 'split'], metrics)
    summary.to_csv(root / 'results_summary.csv', index=False, float_format='%.17g')
    comparison = summary[['lambda', 'split', 'seeds', 'accuracy_mean', 'accuracy_std', 'spearman_hi_cycle_mean']].copy()
    comparison['experiment'] = comparison['lambda'].map(lambda v: f'S2_lambda_{v:g}')
    comparison['source'] = str(root)
    prior = []
    for name, dev, test, count, source in [('OCALD_formal', .5435, .5235, 1, 'user-provided historical formal OCALD'),
                                         ('OCALD_standardized_mu_I', .5826, .5599, 1, 'README D1/D2, seed42'),
                                         ('F1_statistics', .9349, .9409, 5, 'README A/B F1, five-seed mean')]:
        for split, accuracy in [('dev', dev), ('test', test)]:
            prior.append({'experiment': name, 'split': split, 'seeds': count, 'accuracy_mean': accuracy, 'source': source})
    e1 = pd.read_csv(settings['e1_results'], float_precision='round_trip')
    for _, row in e1[e1.variant.isin(['V_C', 'V_ORD'])].iterrows():
        if row['mode'] != 'full' or row.seeds != 3:
            raise ValueError('E1 comparison source is not the three-seed full result')
        prior.append({'experiment': 'E1_'+row.variant, 'split': row.split, 'seeds': row.seeds,
                      'accuracy_mean': row.accuracy_mean, 'accuracy_std': row.accuracy_std,
                      'spearman_hi_cycle_mean': row.spearman_hi_cycle_mean, 'source': settings['e1_results']})
    pd.concat([comparison, pd.DataFrame(prior)], ignore_index=True).to_csv(root / 'comparison.csv', index=False, float_format='%.17g')
    return summary


def run(config, smoke, e1_results):
    validate_hi_config(config)
    if not Path(e1_results).is_file():
        raise ValueError('Missing E1 CSV: supply --e1-results explicitly; no fallback')
    processed = load_processed(config)
    datasets = datasets_for(processed, smoke)
    epochs, batches = (3, 2) if smoke else (100, math.ceil(len(datasets['dev'])/16))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        torch.set_num_threads(1)
    root = Path('outputs/ocald_s2') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root.mkdir(parents=True, exist_ok=False)
    settings = {'mode': 'smoke' if smoke else 'full', 'epochs': epochs, 'probe_epochs': epochs, 'batches_per_epoch': batches,
                'lambdas': list(LAMBDAS), 'seeds': list(SEEDS), 'learning_rate': .001, 'weight_decay': 0, 'clip_norm': 5,
                'l2_coefficient': 1e-4, 'l2_scaled_by_lambda': False, 'pair_count_per_batch': 16, 'kappa': 1,
                'batch_scheme': 'grouped', 'mi_seed': 42, 'device': str(device), 'config': config,
                'e1_results': str(e1_results), 'e1_sha256': file_hash(e1_results), 'protected_sha256': protected_hashes(config),
                'keys': {split: [[f['unit'], f['cycle']] for f in dataset.flights] for split, dataset in datasets.items()}}
    dump_json(root / 'run_settings.json', settings)
    print(f'S2 {settings["mode"]}: device={device}; dev/test={len(datasets["dev"])}/{len(datasets["test"])}; epochs={epochs}; output={root}', flush=True)
    joint = {str(seed): joint_schedule(datasets['dev'].flights, epochs, batches, seed) for seed in SEEDS}
    units = torch.tensor([f['unit'] for f in datasets['dev'].flights], device=device)
    cycles = torch.tensor([f['cycle'] for f in datasets['dev'].flights], device=device)
    probe = {str(seed): pair_schedule(valid_pair_pools(units, cycles, 5), len(units), epochs, seed).cpu() for seed in SEEDS}
    torch.save({'joint': joint, 'probe': probe, 'dev_keys': settings['keys']['dev']}, root / 'sampling_schedule.pt')
    first = joint['42']
    batch = move_batch([datasets['dev'].flights[i] for i in first['flights'][0, 0].tolist()], device)
    checks = check_mechanisms(config, batch, first['pairs'][0, 0].to(device))
    del batch
    initial_hashes, noise_hashes = {}, {}
    for seed in SEEDS:
        for weight in LAMBDAS:
            folder = folder_for(root, weight, seed)
            folder.mkdir(parents=True)
            train_joint(config, datasets['dev'], datasets, joint[str(seed)], weight, seed, epochs, folder, processed, device)
            saved = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
            for hashes, field in ((initial_hashes, 'initial_sha256'), (noise_hashes, 'reparameterization_rng_digest')):
                if seed in hashes:
                    assert hashes[seed] == saved[field]
                hashes[seed] = saved[field]
    probe_rows, ablation_rows = [], []
    for seed in SEEDS:
        for weight in LAMBDAS:
            folder = folder_for(root, weight, seed)
            model, saved = load_s2(folder / 'checkpoint.pt', device)
            ablation_rows.extend(ablate(model, saved['posterior_means'], datasets, weight, seed, device))
            probe_rows.extend(train_probes(model, saved, datasets, probe[str(seed)], folder, device, epochs))
    pd.DataFrame(ablation_rows).to_csv(root / 'latent_ablation.csv', index=False, float_format='%.17g')
    probe_frame = pd.DataFrame(probe_rows)
    probe_frame.to_csv(root / 'probe_results.csv', index=False, float_format='%.17g')
    summarize(probe_frame[probe_frame.unit == 'all'], ['lambda', 'probe', 'split'], ['accuracy', 'order', 'spearman_hi_cycle']).to_csv(root / 'probe_summary.csv', index=False, float_format='%.17g')
    summary = evaluate_run(root, settings, processed, datasets, device)
    if protected_hashes(config) != settings['protected_sha256'] or file_hash(e1_results) != settings['e1_sha256']:
        raise RuntimeError('Existing files/data/comparison source changed')
    checks.update({'joint_initialization_shared_per_seed': True, 'reparameterization_rng_shared_per_seed': True,
                   'probe_representation_frozen': True, 'labels_read_after_all_training': True, 'existing_files_unchanged': True})
    verify_run(root, device)
    checks['checkpoint_prediction_metric_probe_ablation_summary_reload_verified'] = True
    dump_json(root / 'verification.json', checks)
    notes = ['S2 diagnostic: original OCALD architecture and flight-summed ELBO plus encoder-backpropagated ranking.',
             f'Mode={settings["mode"]}; lambdas={list(LAMBDAS)}; seeds={list(SEEDS)}; joint/probe epochs={epochs}.',
             'L=ELBO+lambda*order+1e-4*head_L2; L2 not multiplied by lambda; all-model gradient norm clip=5.',
             '4 distinct units x 4 distinct flights; redraw group with no gap>=5; repeat across batches.',
             '16 pairs per batch: uniform eligible unit then uniform valid pair WITHIN BATCH, not global-dev uniform pairs.',
             'Historical baseline has different batch distribution; no conclusion attributing all differences solely to the loss.',
             'Raw=sigmoid(-HIHead(mu_I)); scaled uses dev-only raw extrema, no direction flip or clipping.',
             'Ranking uses r; signed Spearman uses raw HI; ties are incorrect, units equal; seed std ddof=0.',
             'Probes use frozen I/C/IC, dev-only standardization and newly initialized heads, not additional Stage 2 of the joint model.',
             'HI/RUL read only after all joint and probe training; no test-based lambda/model selection.',
             'Unavailable metric reasons are in metrics.json; zero-range scaled values are unavailable.', '',
             summary[['lambda', 'split', 'accuracy_mean', 'accuracy_std', 'spearman_hi_cycle_mean']].to_string(index=False)]
    (root / 'summary.txt').write_text('\n'.join(notes)+'\n', encoding='utf-8')
    print(summary[['lambda', 'split', 'accuracy_mean', 'accuracy_std', 'spearman_hi_cycle_mean']].to_string(index=False), flush=True)
    print(f'PASS: S2 {settings["mode"]}; output={root}', flush=True)


def verify_run(root, device):
    root = Path(root)
    settings = json.loads((root / 'run_settings.json').read_text())
    config = settings['config']
    if protected_hashes(config) != settings['protected_sha256'] or file_hash(settings['e1_results']) != settings['e1_sha256']:
        raise ValueError('Protected/comparison hashes changed')
    processed = load_processed(config)
    datasets = datasets_for(processed, settings['mode'] == 'smoke')
    schedule = torch.load(root / 'sampling_schedule.pt', map_location='cpu', weights_only=True)
    rank_table = pd.read_csv(root / 'ranking_per_unit.csv', float_precision='round_trip')
    probe_table = pd.read_csv(root / 'probe_results.csv', float_precision='round_trip')
    ablation = pd.read_csv(root / 'latent_ablation.csv', float_precision='round_trip')
    initial_hashes, noise_hashes = {}, {}
    for seed in SEEDS:
        fresh = joint_schedule(datasets['dev'].flights, settings['epochs'], settings['batches_per_epoch'], seed)
        assert all(torch.equal(fresh[key], schedule['joint'][str(seed)][key]) for key in fresh)
        for weight in LAMBDAS:
            folder = folder_for(root, weight, seed)
            model, saved = load_s2(folder / 'checkpoint.pt', device)
            assert saved['epoch'] == settings['epochs'] and saved['seed'] == seed and saved['lambda'] == weight
            assert state_hash(saved['initial_state_dict']) == saved['initial_sha256']
            for hashes, field in ((initial_hashes, 'initial_sha256'), (noise_hashes, 'reparameterization_rng_digest')):
                if seed in hashes:
                    assert hashes[seed] == saved[field]
                hashes[seed] = saved[field]
            logs = pd.read_csv(folder / 'losses.csv', float_precision='round_trip')
            assert logs.epoch.tolist() == list(range(1, settings['epochs']+1))
            assert logs.encoder_order_grad_norm.iloc[0] > 0
            np.testing.assert_allclose(logs.elbo, logs.reconstruction+logs.kl_i+logs.kl_c, rtol=1e-6, atol=1e-5)
            np.testing.assert_allclose(logs.loss, logs.elbo+weight*logs.order+1e-4*logs.l2, rtol=1e-6, atol=1e-5)
            blocks = {split: extract_blocks(model, dataset, 16, device) for split, dataset in datasets.items()}
            predictions = pd.read_csv(folder / 'predictions.csv', float_precision='round_trip')
            reports = json.loads((folder / 'metrics.json').read_text())
            for split, values in blocks.items():
                table = predictions[predictions.split == split].reset_index(drop=True)
                assert table[['unit','cycle']].values.tolist() == settings['keys'][split]
                for name in ('I','C'):
                    torch.testing.assert_close(values[name][0].cpu(), saved['posterior_means'][split][name], rtol=0, atol=0)
                with torch.no_grad():
                    r = model.hi_head(values['I'][0])
                    raw = torch.sigmoid(-r).cpu()
                np.testing.assert_array_equal(r.cpu().numpy(), table.r)
                np.testing.assert_array_equal(raw.numpy(), table.hi_raw)
                scale = saved['hi_minmax']
                if split == 'dev':
                    assert scale['min'] == raw.min().item() and scale['max'] == raw.max().item() and scale['count'] == len(raw)
                if scale['min'] != scale['max']:
                    np.testing.assert_array_equal(scale_hi(raw, scale).numpy(), table.hi_scaled)
                else:
                    assert table.hi_scaled.isna().all()
                for label, column in (('raw','hi_raw'),('scaled','hi_scaled')):
                    assert metric_entry(table, column) == reports[split][label]
                ranks = ranking_rows('joint', split, seed, r, values['I'][1], values['I'][2], 5, 1)
                stored = rank_table[(rank_table['lambda'] == weight) & (rank_table.seed == seed) & (rank_table.split == split)].reset_index(drop=True)
                compare_ranking(ranks, stored)
            fresh_ablation = pd.DataFrame(ablate(model, saved['posterior_means'], datasets, weight, seed, device))
            stored_ablation = ablation[(ablation['lambda'] == weight) & (ablation.seed == seed)].reset_index(drop=True)
            assert fresh_ablation[['split','setting']].equals(stored_ablation[['split','setting']])
            np.testing.assert_allclose(fresh_ablation.reconstruction_mean, stored_ablation.reconstruction_mean, rtol=0, atol=0)
            transforms = {name: fit_transform(blocks['dev'][name][0]) for name in ('I','C')}
            for name in ('I','C','IC'):
                probe_folder = folder / 'probes' / name
                probe = torch.load(probe_folder / 'head.pt', map_location='cpu', weights_only=True)
                assert probe['epoch'] == settings['probe_epochs'] and probe['source_sha256'] == file_hash(folder / 'checkpoint.pt')
                transform = probe['transform']
                for key in ('mean','std','denominator'):
                    expected = transforms[name][key] if name != 'IC' else torch.cat((transforms['I'][key], transforms['C'][key]))
                    torch.testing.assert_close(transform[key], expected, rtol=0, atol=0)
                head = HIHead(probe['input_dim'],16).to(device).eval()
                head.load_state_dict(probe['state_dict'])
                probe_predictions = pd.read_csv(probe_folder / 'predictions.csv', float_precision='round_trip')
                log = pd.read_csv(probe_folder / 'losses.csv', float_precision='round_trip')
                assert log.epoch.tolist() == list(range(1,settings['probe_epochs']+1))
                np.testing.assert_allclose(log.loss, log.order+1e-4*log.l2, rtol=1e-6, atol=1e-7)
                for split, values in blocks.items():
                    raw = values[name][0] if name != 'IC' else torch.cat((values['I'][0], values['C'][0]),1)
                    with torch.no_grad():
                        r = head(apply_transform(raw,transform))
                    disk = probe_predictions[probe_predictions.split == split]
                    np.testing.assert_array_equal(r.cpu().numpy(),disk.r)
                    ranks = ranking_rows(name,split,seed,r,values['I'][1],values['I'][2],5,1)
                    stored = probe_table[(probe_table['lambda']==weight) & (probe_table.seed==seed) & (probe_table.probe==name) & (probe_table.split==split)].reset_index(drop=True)
                    compare_ranking(ranks,stored)
    for source, target, groups in [('results_per_seed.csv','results_summary.csv',['lambda','split']),
                                    ('probe_results.csv','probe_summary.csv',['lambda','probe','split'])]:
        frame = pd.read_csv(root / source,float_precision='round_trip')
        if 'unit' in frame:
            frame = frame[frame.unit=='all']
        summary = pd.read_csv(root / target,float_precision='round_trip')
        metrics = [key[:-5] for key in summary.columns if key.endswith('_mean')]
        recomputed = summarize(frame,groups,metrics)
        assert recomputed[groups].equals(summary[groups])
        for field in recomputed.columns.drop(groups):
            np.testing.assert_allclose(recomputed[field],summary[field],rtol=0,atol=1e-12,equal_nan=True)
    print(f'PASS: S2 saved checkpoints, schedules, losses, predictions, metrics, ablations, probes and summaries; output={root}', flush=True)


def compare_ranking(rows, stored):
    fresh = pd.DataFrame(rows)
    assert fresh.unit.tolist() == stored.unit.tolist()
    for key in ('flights','pairs','correct_count','tie_count','accuracy','order','spearman_hi_cycle'):
        np.testing.assert_allclose(fresh[key],stored[key],rtol=0,atol=1e-14)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--batch-scheme',choices=['grouped'],default='grouped')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--e1-results',default=E1_RESULTS,help='Explicit historical E1 comparison CSV; no fallback')
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--run-dir')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.verify_only:
        if args.run_dir is None or args.smoke:
            parser.error('--verify-only requires --run-dir and excludes --smoke')
        settings = json.loads((Path(args.run_dir)/'run_settings.json').read_text())
        if settings['config'] != config:
            parser.error('Configuration differs from saved run')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type=='cpu':
            torch.set_num_threads(1)
        verify_run(args.run_dir,device)
    else:
        if args.run_dir is not None:
            parser.error('--run-dir is only for read-only verification')
        run(config,args.smoke,args.e1_results)


if __name__=='__main__':
    main()
