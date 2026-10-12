"""Read-only decoder latent ablations on existing DS03 representation weights."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import FlightDataset, collate_fn, load_config, load_processed
from losses import reconstruction_loss
from model import time_positions
from train import freeze_representation, load_representation, representation_state


def load_inputs(config, path, device):
    model, source = load_representation(path, device)
    if source['config'] != config or source['mode'] != 'formal' or source['epoch'] != 100:
        raise ValueError('Expected matching completed formal representation checkpoint')
    processed = load_processed(config)
    if source['columns'] != processed['columns']:
        raise ValueError('Checkpoint/data columns differ')
    for field in ('x', 'w'):
        for key in ('min', 'max'):
            if not torch.equal(source['normalization'][field][key], processed['normalization'][field][key]):
                raise ValueError('Checkpoint/data normalization differs')
    freeze_representation(model)
    return model, processed


def new_output(kind):
    path = Path('outputs/diagnostics') / f"{kind}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def check_unchanged(model, before, path, digest):
    after = representation_state(model)
    if any(not torch.equal(before[key], after[key]) for key in before):
        raise RuntimeError('Representation changed')
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
        raise RuntimeError('Source checkpoint changed')
    for name in ('sensor_encoder', 'conditional_prior', 'decoder'):
        module = getattr(model, name)
        if module.training or any(p.requires_grad or p.grad is not None for p in module.parameters()):
            raise RuntimeError('Representation is not frozen')


def run(config, representation, smoke):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        torch.set_num_threads(1)
    digest = hashlib.sha256(Path(representation).read_bytes()).hexdigest()
    model, processed = load_inputs(config, representation, device)
    before = representation_state(model)
    output = new_output('latent')
    mode = 'smoke' if smoke else 'full'
    print(f'latent {mode}: device={device}; output={output}', flush=True)
    sigma = config['training']['representation']['sigma_x_squared']
    batch_size = config['training']['representation']['batch_size']
    rows, metadata = [], {}
    with torch.no_grad():
        for split in ('dev', 'test'):
            dataset = FlightDataset(processed, split)
            if smoke:
                indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))[:32].tolist()
                dataset.flights = [dataset.flights[i] for i in sorted(indices)]
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            posterior, prior = [], []
            for batch in loader:
                mask = batch['mask'].to(device)
                posterior.append(model.encode(batch['x'].to(device), mask))
                prior.append(model.conditional_prior(batch['w'].to(device), mask)[0])
            mu_i, lv_i, mu_c, lv_c = [torch.cat([p[i] for p in posterior]) for i in range(4)]
            prior_mu = torch.cat(prior)
            rng = torch.Generator(device=device).manual_seed(42)
            zi = mu_i + (0.5 * lv_i).exp() * torch.randn(mu_i.shape, generator=rng, device=device)
            zc = mu_c + (0.5 * lv_c).exp() * torch.randn(mu_c.shape, generator=rng, device=device)
            permutation = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42)).to(device)
            settings = {'posterior_mean': (mu_i, mu_c), 'posterior_sample': (zi, zc),
                        'zero_both': (torch.zeros_like(mu_i), torch.zeros_like(mu_c)),
                        'zero_i': (torch.zeros_like(mu_i), mu_c),
                        'zero_c': (mu_i, torch.zeros_like(mu_c)),
                        'prior_c_zero_i': (torch.zeros_like(mu_i), prior_mu),
                        'shuffled_mean': (mu_i[permutation], mu_c[permutation])}
            totals = {name: 0.0 for name in settings}
            elements, offset = 0, 0
            for batch in loader:
                x, mask = batch['x'].to(device), batch['mask'].to(device)
                count = len(x)
                positions = time_positions(mask, x.dtype)
                elements += int(mask.sum()) * x.shape[-1]
                for name, (latent_i, latent_c) in settings.items():
                    prediction = model.decode(latent_i[offset:offset + count], latent_c[offset:offset + count], positions)
                    losses = reconstruction_loss(x, prediction, mask, sigma)
                    if not torch.isfinite(losses).all():
                        raise RuntimeError(f'Nonfinite reconstruction: {split}/{name}')
                    # Independent valid-element check also checks padding exclusion.
                    sse = (x[mask].double() - prediction[mask].double()).square().sum()
                    torch.testing.assert_close(losses.double().sum() * (2 * sigma), sse, rtol=1e-6, atol=1e-6)
                    totals[name] += losses.double().sum().item()
                offset += count
            assert offset == len(dataset)
            metadata[split] = {'keys': [[f['unit'], f['cycle']] for f in dataset.flights],
                               'permutation': permutation.cpu().tolist(),
                               'unchanged_shuffle_flights': int((permutation == torch.arange(len(dataset), device=device)).sum())}
            for name, total in totals.items():
                rows.append({'setting': name, 'split': split, 'reconstruction_mean': total / len(dataset),
                             'element_mse': total * (2 * sigma) / elements,
                             'flights': len(dataset), 'valid_elements': elements, 'mode': mode})
    check_unchanged(model, before, representation, digest)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'reconstruction.csv', index=False, float_format='%.17g')
    notes = [f'Mode={mode}; device={device}; seed=42; sigma_x_squared={sigma}.',
             'Reconstruction is flight-equal mean; MSE is pooled over all valid sensor elements.',
             'Posterior sampling uses one draw per flight; shuffle moves I/C together within each split.',
             'A small zero-both difference supports weak decoder use of latents on this data; it alone does not prove posterior collapse.']
    for split in ('dev', 'test'):
        table = frame[frame.split == split].set_index('setting')
        base = table.loc['posterior_mean', 'reconstruction_mean']
        for name in table.index:
            delta = table.loc[name, 'reconstruction_mean'] - base
            relative = f'{100 * delta / base:.6f}%' if base != 0 else 'undefined (zero baseline)'
            notes.append(f'{split}/{name}: reconstruction={table.loc[name, "reconstruction_mean"]:.9f}, MSE={table.loc[name, "element_mse"]:.9f}, delta={delta:.9f} ({relative}).')
    (output / 'summary.txt').write_text('\n'.join(notes) + '\n', encoding='utf-8')
    (output / 'verification.json').write_text(json.dumps({'mode': mode, 'device': str(device),
        'source': str(representation), 'sha256': digest, 'seed': 42, 'config': config,
        'frozen_and_source_unchanged': True, 'masked_loss_checked': True, 'splits': metadata}, indent=2), encoding='utf-8')
    print(frame.to_string(index=False), flush=True)
    print(f'PASS: latent {mode}; output={output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--representation', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(load_config(args.config), args.representation, args.smoke)


if __name__ == '__main__':
    main()
