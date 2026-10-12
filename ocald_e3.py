"""Encoder isolation: T1/T2 deterministic ranking, no generative modules."""
import argparse
from datetime import datetime, timezone
import math
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from data import load_config, load_processed, read_reference, read_rul
from diagnose import fit_transform, apply_transform, pair_schedule
from diagnostics_features import ranking_rows
from losses import order_loss, head_l2, hi_loss
from model import SensorEncoder, HIHead, valid_lengths
from ocald_s2 import (datasets_for, move_batch, joint_schedule, dump_json, file_hash,
                      protected_hashes, state_hash, metric_entry)
from train import valid_pair_pools, scale_hi


class ConvEncoder(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.length = length
        self.convs = nn.ModuleList([nn.Conv1d(a, b, 11, padding=5)
                                   for a, b in [(14, 128), (128, 64), (64, 16)]])
        self.mu_i = nn.Linear(16 * length, 8)

    def forward(self, x, mask):
        lengths = valid_lengths(mask)
        if int(lengths.max()) > self.length:
            raise ValueError('Valid flight exceeds fixed convolution length')
        x = x.masked_fill(~mask[:, :, None], 0)
        if x.shape[1] > self.length:
            x, mask = x[:, :self.length], mask[:, :self.length]
        else:
            pad = self.length - x.shape[1]
            x, mask = F.pad(x, (0, 0, 0, pad)), F.pad(mask, (0, pad), value=False)
        state = x.transpose(1, 2)
        for conv in self.convs:
            state = F.relu(conv(state)).masked_fill(~mask[:, None, :], 0)
        return self.mu_i(state.flatten(1))


class RankingModel(nn.Module):
    def __init__(self, variant, length):
        super().__init__()
        self.variant = variant
        self.sensor_encoder = SensorEncoder(14, 64, 8, 8) if variant == 'T1' else ConvEncoder(length)
        self.hi_head = HIHead(8, 16)
        if variant == 'T1':
            # Original SensorEncoder is imported unchanged; unused output heads stay fixed.
            for name in ('logvar_i', 'mu_c', 'logvar_c'):
                for p in getattr(self.sensor_encoder, name).parameters():
                    p.requires_grad_(False)

    def mean(self, x, mask):
        value = self.sensor_encoder(x, mask)
        return value[0] if self.variant == 'T1' else value

    def forward(self, x, mask):
        return self.hi_head(self.mean(x, mask))


def norms(model, gradients):
    values = {'encoder': [], 'backbone': [], 'mu_i': []}
    for (name, _), grad in zip([(n, p) for n, p in model.sensor_encoder.named_parameters() if p.requires_grad], gradients):
        if grad is None:
            raise RuntimeError(f'Missing encoder ranking gradient: {name}')
        if not torch.isfinite(grad).all():
            raise RuntimeError('Nonfinite encoder gradient')
        square = grad.detach().double().square().sum().item()
        values['encoder'].append(square)
        values['mu_i' if name.startswith('mu_i.') else 'backbone'].append(square)
    return {k: math.sqrt(sum(v)) for k, v in values.items()}


def extract(model, dataset, device):
    model.eval()
    values, scores = [], []
    with torch.no_grad():
        for start in range(0, len(dataset), 16):
            batch = move_batch(dataset.flights[start:start+16], device)
            mean = model.mean(batch['x'], batch['mask'])
            values.append(mean.detach())
            scores.append(model.hi_head(mean).detach())
    return torch.cat(values), torch.cat(scores)


def rank(name, split, seed, scores, dataset):
    device = scores.device
    units = torch.tensor([f['unit'] for f in dataset.flights], device=device)
    cycles = torch.tensor([f['cycle'] for f in dataset.flights], device=device)
    return ranking_rows(name, split, seed, scores, units, cycles, 5, 1)


def mechanisms(model, batch, pairs):
    model.train()  # cuDNN GRU backward requires train-mode forward.
    scores = model(batch['x'], batch['mask'])
    loss = order_loss(scores[pairs[:, 0]], scores[pairs[:, 1]], 1)
    params = [p for p in model.sensor_encoder.parameters() if p.requires_grad]
    check = norms(model, torch.autograd.grad(loss, params))
    assert check['encoder'] > 0 and check['backbone'] > 0 and check['mu_i'] > 0
    mask = batch['mask'].clone()
    model.eval()
    with torch.no_grad():
        mean = model.mean(batch['x'], mask)
        changed = batch['x'].masked_fill(~mask[:, :, None], float('nan'))
        torch.testing.assert_close(mean, model.mean(changed, mask), rtol=1e-6, atol=1e-6)
        x = F.pad(changed, (0, 0, 0, 13), value=999)
        extended = F.pad(mask, (0, 13), value=False)
        torch.testing.assert_close(mean, model.mean(x, extended), rtol=1e-6, atol=1e-6)
    assert torch.equal(mask, batch['mask'])
    assert order_loss(torch.tensor([0.]), torch.tensor([1.]), 1) < order_loss(torch.tensor([1.]), torch.tensor([0.]), 1)
    return {'mask_checked': True, 'initial_order_gradient_norms': check}


def plot_training(frame, folder, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    axes[0].plot(frame.epoch, frame.order)
    axes[0].set_ylabel('Training order loss')
    valid = frame.dropna(subset=['dev_accuracy'])
    axes[1].plot(valid.epoch, valid.dev_accuracy * 100, marker='o')
    axes[1].set_ylabel('Dev exhaustive accuracy (%)')
    for field in ('encoder_order_grad_norm', 'encoder_total_preclip_grad_norm', 'encoder_total_postclip_grad_norm'):
        axes[2].plot(frame.epoch, frame[field], label=field)
    axes[2].set_ylabel('Mean encoder gradient norm')
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.grid(alpha=.25)
    fig.suptitle(title)
    fig.savefig(folder/'training_curves.png', dpi=140)
    plt.close(fig)


def train_model(variant, seed, length, epochs, schedule, datasets, folder, device):
    torch.manual_seed(seed)
    model = RankingModel(variant, length).to(device)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    first = schedule['flights'][0, 0]
    batch = move_batch([datasets['dev'].flights[i] for i in first.tolist()], device)
    checks = mechanisms(model, batch, schedule['pairs'][0, 0].to(device))
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=0)
    params = [p for p in model.sensor_encoder.parameters() if p.requires_grad]
    logs = []
    for epoch in range(epochs):
        model.train()
        steps = []
        for indices, cpu_pairs in zip(schedule['flights'][epoch], schedule['pairs'][epoch]):
            batch = move_batch([datasets['dev'].flights[i] for i in indices.tolist()], device)
            pairs = cpu_pairs.to(device)
            optimizer.zero_grad(set_to_none=True)
            scores = model(batch['x'], batch['mask'])
            order = order_loss(scores[pairs[:, 0]], scores[pairs[:, 1]], 1)
            l2 = head_l2(model.hi_head)
            loss = order + 1e-4 * l2
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            grad = norms(model, torch.autograd.grad(order, params, retain_graph=True))
            loss.backward()
            pre = norms(model, [p.grad for p in params])
            total = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True).item()
            post = norms(model, [p.grad for p in params])
            optimizer.step()
            steps.append({'loss': loss.item(), 'order': order.item(), 'l2': l2.item(),
                          'encoder_order_grad_norm': grad['encoder'], 'backbone_order_grad_norm': grad['backbone'],
                          'mu_i_order_grad_norm': grad['mu_i'], 'encoder_total_preclip_grad_norm': pre['encoder'],
                          'encoder_total_postclip_grad_norm': post['encoder'], 'total_preclip_grad_norm': total,
                          'zero_order_gradient': int(grad['encoder'] == 0)})
        log = {'epoch': epoch+1, **pd.DataFrame(steps).mean().to_dict(), 'dev_accuracy': np.nan}
        if epoch == 0 or (epoch+1) % 10 == 0 or epoch+1 == epochs:
            _, r = extract(model, datasets['dev'], device)
            log['dev_accuracy'] = rank(variant, 'dev', seed, r, datasets['dev'])[-1]['accuracy']
            print(f'{variant} seed={seed} epoch={epoch+1}/{epochs} order={log["order"]:.6f} dev={100*log["dev_accuracy"]:.2f}% gradient={log["encoder_order_grad_norm"]:.6g}', flush=True)
        logs.append(log)
        pd.DataFrame(logs).to_csv(folder/'losses.csv', index=False, float_format='%.17g')
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert any(not torch.equal(state[k], v) for k, v in initial.items() if k.startswith('sensor_encoder.'))
    if variant == 'T1':
        for k in initial:
            if k.startswith(('sensor_encoder.logvar_', 'sensor_encoder.mu_c.')):
                assert torch.equal(initial[k], state[k])
    blocks = {split: extract(model, dataset, device) for split, dataset in datasets.items()}
    raw = torch.sigmoid(-blocks['dev'][1]).cpu()
    scaling = {'min': raw.min().item(), 'max': raw.max().item(), 'fit_split': 'dev', 'count': len(raw)}
    saved = {'stage': 'ocald_e3', 'variant': variant, 'seed': seed, 'epoch': epochs, 'length': length,
             'state_dict': state, 'initial_state_dict': initial, 'hi_minmax': scaling,
             'means': {s: v[0].cpu() for s, v in blocks.items()}, 'scores': {s: v[1].cpu() for s, v in blocks.items()}}
    torch.save(saved, folder/'checkpoint.pt')
    dump_json(folder/'verification.json', checks)
    plot_training(pd.DataFrame(logs), folder, f'E3 {variant} | seed={seed}')
    return model, saved


def probe(model, saved, datasets, schedule, folder, device):
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None
    model.eval()
    before = state_hash(model.state_dict())
    transform = fit_transform(saved['means']['dev'])
    inputs = {s: apply_transform(v.to(device), transform) for s, v in saved['means'].items()}
    torch.manual_seed(saved['seed'])
    head = HIHead(8, 16).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=.001, weight_decay=0)
    logs = []
    for epoch, pairs in enumerate(schedule, 1):
        steps = []
        for start in range(0, len(pairs), 128):
            pair = pairs[start:start+128].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = hi_loss(head(inputs['dev'][pair[:, 0]]), head(inputs['dev'][pair[:, 1]]), head, 1, 1e-4)
            loss['loss'].backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
            optimizer.step()
            steps.append({k: v.item()*len(pair) for k, v in loss.items()})
        logs.append({'epoch': epoch, **(pd.DataFrame(steps).sum()/len(pairs)).to_dict()})
    folder.mkdir()
    pd.DataFrame(logs).to_csv(folder/'losses.csv', index=False, float_format='%.17g')
    with torch.no_grad():
        scores = {s: head(x).cpu() for s, x in inputs.items()}
    torch.save({'state_dict': head.state_dict(), 'transform': transform, 'scores': scores,
                'seed': saved['seed'], 'epoch': len(schedule)}, folder/'head.pt')
    rows = [row for s, r in scores.items() for row in rank(saved['variant']+'_probe_I', s, saved['seed'], r.to(device), datasets[s])]
    pd.DataFrame(rows).to_csv(folder/'ranking.csv', index=False, float_format='%.17g')
    assert before == state_hash(model.state_dict())
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    return rows


def plot_hi(table, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = list(table.groupby('unit', sort=True))
    fig, axes = plt.subplots(len(groups), 2, figsize=(14, 2.5*len(groups)), layout='constrained', squeeze=False)
    fig.suptitle(title)
    for row, (unit, group) in enumerate(groups):
        group = group.sort_values('cycle')
        for col, field in enumerate(('hi_raw', 'hi_scaled')):
            ax = axes[row, col]
            ax.plot(group.cycle, group.hi_gt, color='black', linewidth=1, label='Reference HI')
            ax.plot(group.cycle, group[field], linewidth=1.1, label=field)
            ax.set_title(f'unit {unit} | {field}')
            ax.set_xlabel('Flight cycle')
            ax.set_ylabel('HI')
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_summary(root):
    settings = json.loads((root/'run_settings.json').read_text())
    results = pd.read_csv(root/'results.csv')
    lines = [f'E3 mode={"smoke" if settings["smoke"] else "full"}; epochs={settings["epochs"]}; device={settings["device"]}; length={settings["length"]}',
             'Only T1/T2 ranking+head L2; no decoder/prior/ELBO/KL/sampling.',
             'Fixed max length from processed shapes; per-layer convolution masking; default initialization.',
             'Unit/cycle only sampling/evaluation. Labels first read after all training. No early stop or test selection.']
    for variant in settings['variants']:
        for seed in settings['seeds']:
            log = pd.read_csv(root/variant/f'seed_{seed}'/'losses.csv')
            lines.append(f'{variant} seed={seed}: training order first={log.order.iloc[0]:.8f} last={log.order.iloc[-1]:.8f}; '
                         f'mean encoder order gradient range={log.encoder_order_grad_norm.min():.8g}..{log.encoder_order_grad_norm.max():.8g}; '
                         f'last={log.encoder_order_grad_norm.iloc[-1]:.8g}; zero-gradient batch fractions max={log.zero_order_gradient.max():.8g}.')
            table = pd.read_csv(root/variant/f'seed_{seed}'/'predictions.csv')
            if (table.hi_raw == 0).all():
                lines.append(f'{variant}: raw Sigmoid HI is zero for all flights; existing metrics convention returns signed Spearman=0 for constants; scaled HI unavailable. Ranking still uses r.')
    lines.extend(f'{r.variant} seed={r.seed} {r.split}: accuracy={r.accuracy*100:.4f}% signed Spearman={r.spearman_hi_cycle:.6f}' for r in results.itertuples())
    (root/'summary.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)


def verify(root, device):
    if device.type == 'cpu':
        torch.set_num_threads(1)
    settings = json.loads((root/'run_settings.json').read_text())
    datasets = datasets_for(load_processed(settings['config']), settings['smoke'])
    assert protected_hashes(settings['config']) == settings['protected_sha256']
    assert file_hash('ocald_s2.py') == settings['s2_sha256']
    assert file_hash(settings['s2_results']) == settings['s2_results_sha256']
    frame = pd.read_csv(root/'results.csv', float_precision='round_trip')
    schedule = torch.load(root/'sampling_schedule.pt', map_location='cpu', weights_only=True)
    units = torch.tensor([f['unit'] for f in datasets['dev'].flights])
    cycles = torch.tensor([f['cycle'] for f in datasets['dev'].flights])
    for seed in settings['seeds']:
        plan = schedule['joint'][str(seed)]
        for indices, pairs in zip(plan['flights'].flatten(0,1), plan['pairs'].flatten(0,1)):
            assert len(indices.unique()) == 16
            assert units[indices].unique(return_counts=True)[1].tolist() == [4]*4
            assert (units[indices[pairs[:,0]]] == units[indices[pairs[:,1]]]).all()
            assert (cycles[indices[pairs[:,1]]] - cycles[indices[pairs[:,0]]] >= 5).all()
        for variant in settings['variants']:
            folder = root/variant/f'seed_{seed}'
            saved = torch.load(folder/'checkpoint.pt', map_location='cpu', weights_only=True)
            assert saved['epoch'] == settings['epochs']
            model = RankingModel(variant, saved['length']).to(device).eval()
            model.load_state_dict(saved['state_dict'])
            log = pd.read_csv(folder/'losses.csv', float_precision='round_trip')
            assert log.epoch.tolist() == list(range(1, settings['epochs']+1))
            np.testing.assert_allclose(log.loss, log.order+1e-4*log.l2, rtol=1e-6)
            assert np.isfinite(log.encoder_order_grad_norm).all() and log.encoder_order_grad_norm.iloc[0] > 0
            head_saved = torch.load(folder/'probe_I/head.pt', map_location='cpu', weights_only=True)
            assert head_saved['epoch'] == settings['epochs']
            transform = fit_transform(saved['means']['dev'])
            for key in ('mean', 'std', 'denominator'):
                torch.testing.assert_close(transform[key], head_saved['transform'][key], rtol=0, atol=0)
            probe_log = pd.read_csv(folder/'probe_I/losses.csv', float_precision='round_trip')
            assert probe_log.epoch.tolist() == list(range(1, settings['epochs']+1))
            np.testing.assert_allclose(probe_log.loss, probe_log.order+1e-4*probe_log.l2, rtol=1e-6)
            probe_rank = pd.read_csv(folder/'probe_I/ranking.csv', float_precision='round_trip', dtype={'unit': str})
            predictions = pd.read_csv(folder/'predictions.csv', float_precision='round_trip')
            reports = json.loads((folder/'metrics.json').read_text())
            head = HIHead(8,16).to(device).eval()
            head.load_state_dict(head_saved['state_dict'])
            for split, dataset in datasets.items():
                means, scores = extract(model, dataset, device)
                torch.testing.assert_close(means.cpu(), saved['means'][split], rtol=0, atol=0)
                torch.testing.assert_close(scores.cpu(), saved['scores'][split], rtol=0, atol=0)
                row = rank(variant, split, seed, scores, dataset)[-1]
                disk = frame[(frame.variant==variant)&(frame.seed==seed)&(frame.split==split)].iloc[0]
                for k in ('accuracy','order','spearman_hi_cycle'):
                    np.testing.assert_allclose(row[k], disk[k], rtol=1e-7, atol=1e-9)
                table = predictions[predictions.split == split].reset_index(drop=True)
                assert list(zip(table.unit, table.cycle)) == [(f['unit'], f['cycle']) for f in dataset.flights]
                np.testing.assert_array_equal(table.r, saved['scores'][split].numpy().astype(float))
                raw = torch.sigmoid(-saved['scores'][split])
                np.testing.assert_array_equal(table.hi_raw, raw.numpy().astype(float))
                scaling = saved['hi_minmax']
                if scaling['min'] != scaling['max']:
                    np.testing.assert_array_equal(table.hi_scaled, scale_hi(raw, scaling).numpy().astype(float))
                else:
                    assert table.hi_scaled.isna().all()
                for scale, column in [('raw', 'hi_raw'), ('scaled', 'hi_scaled')]:
                    assert metric_entry(table, column) == reports[split][scale]
                with torch.no_grad():
                    r = head(apply_transform(means, head_saved['transform'])).cpu()
                torch.testing.assert_close(r, head_saved['scores'][split], rtol=0, atol=0)
                stored = probe_rank[(probe_rank.split == split) & (probe_rank.unit == 'all')].iloc[0]
                probe_row = rank(variant+'_probe_I', split, seed, r.to(device), dataset)[-1]
                for key in ('accuracy', 'order', 'spearman_hi_cycle'):
                    np.testing.assert_allclose(probe_row[key], stored[key], rtol=1e-7, atol=1e-9)
    print(f'PASS: E3 reload, ranking, probes, losses, schedules and protected files; output={root}', flush=True)


def run(args):
    config = load_config(args.config)
    processed = load_processed(config)
    datasets = datasets_for(processed, args.smoke)
    length = max(len(f['x']) for s in ('dev','test') for f in processed[s])
    epochs, batches = (3,2) if args.smoke else (100, math.ceil(len(datasets['dev'])/16))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu': torch.set_num_threads(1)
    root = Path('outputs/ocald_e3')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root.mkdir(parents=True, exist_ok=False)
    settings = {'config': config, 'variants': args.variants, 'seeds': args.seeds, 'smoke': args.smoke,
                'epochs': epochs, 'batches_per_epoch': batches, 'length': length, 'device': str(device),
                'protected_sha256': protected_hashes(config), 's2_sha256': file_hash('ocald_s2.py'),
                's2_results': args.s2_results, 's2_results_sha256': file_hash(args.s2_results)}
    dump_json(root/'run_settings.json', settings)
    joint = {str(s): joint_schedule(datasets['dev'].flights, epochs, batches, s) for s in args.seeds}
    u = torch.tensor([f['unit'] for f in datasets['dev'].flights], device=device)
    c = torch.tensor([f['cycle'] for f in datasets['dev'].flights], device=device)
    probes = {str(s): pair_schedule(valid_pair_pools(u,c,5), len(u), epochs,s).cpu() for s in args.seeds}
    torch.save({'joint':joint, 'probe':probes},root/'sampling_schedule.pt')
    print(f'E3 device={device} dev/test={len(datasets["dev"])}/{len(datasets["test"])} length={length} output={root}', flush=True)
    main_rows, probe_rows = [], []
    for variant in args.variants:
        for seed in args.seeds:
            folder = root/variant/f'seed_{seed}'
            folder.mkdir(parents=True)
            model, saved = train_model(variant,seed,length,epochs,joint[str(seed)],datasets,folder,device)
            probe_rows.extend(probe(model,saved,datasets,probes[str(seed)],folder/'probe_I',device))
            main_rows.extend(row for s,r in saved['scores'].items() for row in rank(variant,s,seed,r.to(device),datasets[s]))
    # Evaluation labels are first read after every main model and probe is trained.
    references = {}
    for split in ('dev','test'):
        refs = read_reference(config['data'][f'hi_{split}_path'],processed[split])
        rul = read_rul(config['data']['raw_path'],split,processed[split])
        references[split] = refs.merge(rul,on=['unit','cycle'],validate='one_to_one').set_index(['unit','cycle'])
    for variant in args.variants:
        for seed in args.seeds:
            folder = root/variant/f'seed_{seed}'
            saved = torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
            tables, reports = [], {}
            for split,dataset in datasets.items():
                keys = pd.MultiIndex.from_tuples([(f['unit'],f['cycle']) for f in dataset.flights],names=['unit','cycle'])
                table = references[split].loc[keys].reset_index()
                r = saved['scores'][split]
                raw = torch.sigmoid(-r)
                table['split'],table['r'],table['hi_raw'] = split,r.numpy().astype(float),raw.numpy().astype(float)
                scale = saved['hi_minmax']
                table['hi_scaled'] = scale_hi(raw,scale).numpy().astype(float) if scale['min']!=scale['max'] else np.nan
                reports[split] = {s:metric_entry(table,k) for s,k in [('raw','hi_raw'),('scaled','hi_scaled')]}
                plot_hi(table,folder/f'hi_curves_{split}.png',f'E3 {variant} | seed={seed} | {split}')
                tables.append(table)
            pd.concat(tables).to_csv(folder/'predictions.csv',index=False,float_format='%.17g')
            dump_json(folder/'metrics.json',reports)
    rankings = pd.DataFrame(main_rows).rename(columns={'feature_set':'variant'})
    rankings.to_csv(root/'ranking_per_unit.csv',index=False,float_format='%.17g')
    results = rankings[rankings.unit=='all'].copy()
    results.to_csv(root/'results.csv',index=False,float_format='%.17g')
    pd.DataFrame(probe_rows).to_csv(root/'probe_results.csv',index=False,float_format='%.17g')
    compare = results[['variant','seed','split','accuracy','spearman_hi_cycle']].copy()
    compare['source'] = str(root)
    prior = []
    for name,dev,test,source in [('OCALD_formal',.5435,.5235,'user-provided formal baseline'),('OCALD_standardized_mu_I',.5826,.5599,'D1/D2 seed42'),('E1_V_ORD',.9634,.9545,'user-provided E1 three-seed mean'),('F1',.9349,.9409,'A/B five-seed mean')]:
        for split,value in [('dev',dev),('test',test)]:
            prior.append({'variant':name,'split':split,'accuracy':value,'source':source})
    s2 = pd.read_csv(args.s2_results)
    for _,row in s2.iterrows():
        prior.append({'variant':f'S2_lambda_{row["lambda"]:g}','split':row.split,'accuracy':row.accuracy_mean,'accuracy_std':row.accuracy_std,'source':args.s2_results+' (three-seed mean)'})
    pd.concat([compare,pd.DataFrame(prior)],ignore_index=True).to_csv(root/'comparison.csv',index=False,float_format='%.17g')
    verify(root,device)
    dump_json(root/'verification.json',{'passed':True,'protected_files_unchanged':True})
    write_summary(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config.yaml')
    parser.add_argument('--variants',nargs='+',choices=['T1','T2'],default=['T1','T2'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[42])
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--s2-results',default='outputs/ocald_s2/20261012T104802398175Z/results_summary.csv')
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--run-dir',type=Path)
    args = parser.parse_args()
    if args.verify_only:
        if args.run_dir is None: parser.error('--verify-only requires --run-dir')
        verify(args.run_dir,torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    else:
        run(args)


if __name__ == '__main__':
    main()
