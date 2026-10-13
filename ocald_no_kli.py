"""Single-seed S2 ablation: zero KL_I training weight, KL_C retained."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from data import FlightDataset, collate_fn, load_config, load_processed
from evaluate import evaluate, predict_table, metric_report
from losses import representation_loss, reconstruction_loss, order_loss, head_l2
from model import OCALD, time_positions
from train import sample_pairs, valid_pair_pools
from torch.utils.data import DataLoader
from torch.nn import functional as F

DEFAULT_S2 = None  # Historical source removed by explicit experiment cleanup.
METRICS = ('mae', 'mape', 'monotonicity', 'trend', 'failure_consistency', 'mi')


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def state_hash(state):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in state.values())).hexdigest()

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

def extract_blocks(model, dataset, batch_size, device):
    """Take I/C from the SAME encode call and check every stored flight key."""
    blocks, units, cycles = {"I": [], "C": []}, [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    with torch.no_grad():
        for batch in loader:
            mu_i, _, mu_c, _ = model.encode(batch["x"].to(device), batch["mask"].to(device))
            for block, values in (("I", mu_i), ("C", mu_c)):
                assert values.shape == (len(batch["unit"]), 8) and torch.isfinite(values).all()
                blocks[block].append(values)
            units.append(batch["unit"].to(device))
            cycles.append(batch["cycle"].to(device))
    units, cycles = torch.cat(units), torch.cat(cycles)
    assert units.cpu().tolist() == [flight["unit"] for flight in dataset.flights]
    assert cycles.cpu().tolist() == [flight["cycle"] for flight in dataset.flights]
    return {block: (torch.cat(values), units, cycles) for block, values in blocks.items()}


def protected(config):
    files = ['model.py', 'losses.py', 'train.py', 'metrics.py', 'evaluate.py', 'data.py',
             'verify.py', 'config.yaml', 'AGENTS.md', 'OCALD_Implementation_Plan.md',
             config['data']['processed_path']]
    notebook = Path('baseline/03_train_evaluate_proposals.ipynb')
    if notebook.exists():
        files.append(str(notebook))
    return {path: file_hash(path) for path in files}


RETIRED_HASH_FILES = {'diagnose.py', 'diagnostics_features.py', 'diagnostics_latent.py',
                      'baseline_ref.py', 'README_baseline_ref.md', 'ocald_s2.py',
                      'README_ocald_s2.md'}


def objective(terms, order, l2):
    # KL_I remains computed for reporting but is absent from this graph.
    return terms['reconstruction'] + terms['kl_c'] + 10 * order + 1e-4 * l2


def check_objective(model, output, batch, pairs):
    terms = representation_loss(batch['x'], batch['mask'], output, 1.)
    r = model.hi_head(output['mu_i'])
    order = order_loss(r[pairs[:, 0]], r[pairs[:, 1]], 1.)
    l2 = head_l2(model.hi_head)
    loss = objective(terms, order, l2)
    grad_i, grad_c = torch.autograd.grad(loss, [terms['kl_i'], terms['kl_c']], allow_unused=True, retain_graph=True)
    assert grad_i is None
    torch.testing.assert_close(grad_c, torch.ones_like(grad_c), rtol=0, atol=0)
    grads = torch.autograd.grad(terms['kl_c'], tuple(model.conditional_prior.parameters()), retain_graph=True)
    prior_norm = torch.stack([g.double().square().sum() for g in grads]).sum().sqrt().item()
    assert np.isfinite(prior_norm) and prior_norm > 0
    ranking_norm = encoder_order_norm(model, order)
    assert ranking_norm > 0
    return {'kl_i_objective_gradient_absent': True, 'kl_c_objective_gradient': grad_c.item(),
            'kl_c_prior_gradient_norm': prior_norm, 'initial_encoder_order_gradient_norm': ranking_norm}


def compare(root, source):
    rows = []
    sources = [('no_kli', root/'metrics.json'), ('S2_lambda10_seed42', root/'s2_reference/metrics.json')]
    for experiment, path in sources:
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        for split in ('dev', 'test'):
            for scale in ('raw', 'scaled'):
                entry = report['splits'][split][scale]
                row = {'experiment': experiment, 'source': str(path), 'split': split, 'scale': scale,
                       'available': entry['available'], **report['splits'][split]['saturation']}
                for metric in METRICS:
                    value = entry.get('overall', {}).get(metric, {})
                    row[metric] = value.get('value')
                    row[metric+'_reason'] = value.get('reason') if entry['available'] else entry['reason']
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(root/'comparison.csv', index=False, float_format='%.17g')
    return frame


def verify(root, device):
    root = Path(root)
    settings = json.loads((root/'run_settings.json').read_text())
    assert protected(settings['config']) == {k: v for k, v in settings['protected_sha256'].items()
                                               if k not in RETIRED_HASH_FILES}
    saved = torch.load(root/'checkpoint.pt', map_location='cpu', weights_only=True)
    assert saved['stage'] == 'ocald_no_kli' and saved['epoch'] == settings['epochs']
    assert saved['actual_training']['kl_i_weight'] == 0 and saved['actual_training']['kl_c_weight'] == 1
    logs = pd.read_csv(root/'losses.csv', float_precision='round_trip')
    assert logs.epoch.tolist() == list(range(1, settings['epochs']+1))
    np.testing.assert_allclose(logs.loss, logs.reconstruction+logs.kl_c+10*logs.order+1e-4*logs.l2, rtol=1e-6, atol=1e-5)
    assert np.isfinite(logs.select_dtypes('number').to_numpy()).all()
    assert logs.encoder_order_grad_norm.iloc[0] > 0
    assert state_hash(saved['initial_state_dict']) == saved['initial_sha256']
    processed = load_processed(settings['config'])
    datasets = datasets_for(processed, settings['mode']=='smoke')
    schedule = torch.load(root/'sampling_schedule.pt', map_location='cpu', weights_only=True)
    fresh = joint_schedule(datasets['dev'].flights, settings['epochs'], settings['batches_per_epoch'], 42)
    assert all(torch.equal(schedule[k],fresh[k]) for k in fresh)
    old_path = Path(settings['s2_checkpoint']) if settings.get('s2_checkpoint') else None
    source_available = old_path is not None and old_path.is_file()
    if source_available:
        assert file_hash(old_path) == settings['s2_sha256']
        old = torch.load(old_path, map_location='cpu', weights_only=True)
        assert saved['initial_sha256'] == old['initial_sha256']
        if settings['mode'] == 'full':
            assert saved['reparameterization_rng_digest'] == old['reparameterization_rng_digest']
            old_schedule = torch.load(old_path.parents[2]/'sampling_schedule.pt',map_location='cpu',weights_only=True)['joint']['42']
            assert all(torch.equal(old_schedule[k],schedule[k]) for k in fresh)
    first = schedule['flights'][0,0]
    pairs = schedule['pairs'][0,0].to(device)
    batch = move_batch([datasets['dev'].flights[i] for i in first.tolist()],device)
    model = OCALD(settings['config']).to(device).train()
    model.load_state_dict(saved['initial_state_dict'])
    checks = check_objective(model, model(batch['x'],batch['w'],batch['mask']), batch, pairs)
    table, loaded = predict_table(settings['config'],root/'checkpoint.pt')
    stored = pd.read_csv(root/'predictions.csv',float_precision='round_trip')
    assert table[['split','unit','cycle']].equals(stored[['split','unit','cycle']])
    for field in ('hi_raw','hi_scaled','hi_gt','rul'):
        np.testing.assert_allclose(table[field],stored[field],rtol=0,atol=0,equal_nan=True)
    assert metric_report(stored,loaded) == json.loads((root/'metrics.json').read_text())
    if (root/'s2_reference').is_dir():
        old_export = pd.read_csv(root/'s2_reference/predictions.csv', float_precision='round_trip')
        old_report = json.loads((root/'s2_reference/metrics.json').read_text())
        if source_available:
            old_table, old_checkpoint = predict_table(settings['config'], old_path)
            assert old_table[['split','unit','cycle']].equals(old_export[['split','unit','cycle']])
            for field in ('hi_raw','hi_scaled','hi_gt','rul'):
                np.testing.assert_allclose(old_table[field], old_export[field], rtol=0, atol=0, equal_nan=True)
        else:
            # Retained comparison report/CSV belong to this run; source was explicitly deleted.
            old_checkpoint = {'stage': old_report['diagnostic_type'], 'mode': old_report['mode'],
                              'epoch': old_report['joint_epoch'], 'seed': old_report['seed'],
                              'lambda': old_report['actual_training']['lambda'],
                              'actual_training': old_report['actual_training'], 'hi_minmax': old_report['hi_minmax']}
            assert not old_export.duplicated(['split','unit','cycle']).any()
            assert old_export[['split','unit','cycle']].equals(stored[['split','unit','cycle']])
            for field in ('hi_gt','rul'):
                np.testing.assert_array_equal(old_export[field],stored[field])
        assert metric_report(old_export, old_checkpoint) == old_report
    comparison = pd.read_csv(root/'comparison.csv', float_precision='round_trip')
    for row in comparison.itertuples():
        entry = json.loads(Path(row.source).read_text())['splits'][row.split][row.scale]
        for key in METRICS:
            value = entry.get('overall', {}).get(key, {}).get('value')
            if value is None:
                assert pd.isna(getattr(row,key))
            else:
                assert value == getattr(row,key)
    dev = stored[stored.split=='dev']
    assert loaded['hi_minmax'] == {'min': dev.hi_raw.min(), 'max': dev.hi_raw.max(), 'count':len(dev), 'fit_split':'dev'}
    for path in root.rglob('*'):
        if path.suffix in ('.json','.csv','.txt','.yaml'):
            text = path.read_text().lower()
            assert 'accuracy' not in text and 'correct_count' not in text, f'Unexpected ranking output: {path}'
    checks.update({'reload_predictions_labels_metrics_scaling_verified': True, 'loss_composition_verified': True,
                   'no_ranking_or_probe_metrics': True, 'protected_files_unchanged': True,
                   'sampling_schedule_recreated_exactly': True,
                   's2_evaluation_and_comparison_recomputed': (root/'s2_reference').is_dir(),
                   'original_s2_checkpoint_available': source_available})
    # Keep completed outputs intact during read-only verification.
    print(f'PASS: no-KL_I loss/gradient/mask/schedule/checkpoint/evaluation checks; output={root}',flush=True)
    return checks


def run(config, smoke, source):
    processed = load_processed(config)
    datasets = datasets_for(processed,smoke)
    epochs,batches = (3,2) if smoke else (100,math.ceil(len(datasets['dev'])/16))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cpu': torch.set_num_threads(1)
    root = Path('outputs/ocald_no_kli')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root.mkdir(parents=True,exist_ok=False)
    source = Path(source) if source else None
    source_available = source is not None and source.is_file()
    actual = {'optimizer':'Adam','lr':.001,'weight_decay':0.,'epochs':epochs,'batch_size':16,
              'gradient_clip_norm':5.,'kl_i_weight':0.,'kl_c_weight':1.,'sigma_x_squared':1.,
              'lambda':10.,'seed':42,'head_l2_coefficient':1e-4,'kappa':1.,'min_cycle_gap':5,
              'pairs_per_batch':16,'batch_scheme':'grouped','checkpoint_selection':'last_epoch',
              'reparameterization':'I/C standard torch.randn_like; same rule as S2'}
    settings = {'config':config,'actual_training':actual,'mode':'smoke' if smoke else 'full',
                'epochs':epochs,'batches_per_epoch':batches,'device':str(device),'protected_sha256':protected(config),
                's2_checkpoint':str(source) if source is not None else None,'s2_sha256':file_hash(source) if source_available else None}
    dump_json(root/'run_settings.json',settings)
    (root/'actual_config.yaml').write_text(yaml.safe_dump({'base_config':config,'diagnostic_type':'ocald_no_kli',
                                                       'mode':settings['mode'],'actual_training':actual},sort_keys=False))
    schedule = joint_schedule(datasets['dev'].flights,epochs,batches,42)
    torch.save(schedule,root/'sampling_schedule.pt')
    if source_available:
        old = torch.load(source,map_location='cpu',weights_only=True)
        assert old['stage']=='ocald_s2' and old['seed']==42 and old['lambda']==10
        if not smoke:
            old_schedule = torch.load(source.parents[2]/'sampling_schedule.pt',map_location='cpu',weights_only=True)['joint']['42']
            assert all(torch.equal(old_schedule[k],schedule[k]) for k in schedule)
    first = schedule['flights'][0,0]
    batch = move_batch([datasets['dev'].flights[i] for i in first.tolist()],device)
    # Mechanism checks run before reseeding training; they cannot consume training noise.
    checks = check_mechanisms(config,batch,schedule['pairs'][0,0].to(device))
    torch.manual_seed(42)
    model = OCALD(config).to(device)
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if source_available: assert state_hash(initial)==old['initial_sha256']
    optimizer = torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0)
    logs=[]
    print(f'no-KL_I {settings["mode"]}: device={device}; epochs={epochs}; output={root}',flush=True)
    for epoch in range(epochs):
        model.train()
        totals = dict.fromkeys(('loss','reconstruction','kl_i','kl_c','order','l2'),0.)
        clips=[]
        order_norm=None
        for step,(indices,cpu_pairs) in enumerate(zip(schedule['flights'][epoch],schedule['pairs'][epoch])):
            batch = move_batch([datasets['dev'].flights[i] for i in indices.tolist()],device)
            pairs=cpu_pairs.to(device)
            optimizer.zero_grad(set_to_none=True)
            output=model(batch['x'],batch['w'],batch['mask'])
            terms=representation_loss(batch['x'],batch['mask'],output,1.)
            r=model.hi_head(output['mu_i'])
            order=order_loss(r[pairs[:,0]],r[pairs[:,1]],1.)
            l2=head_l2(model.hi_head)
            loss=objective(terms,order,l2)
            if not all(torch.isfinite(v) for v in (*terms.values(),order,l2,loss)):
                raise RuntimeError('Nonfinite ablation loss; no silent numerical fallback')
            if step==0:
                order_norm=encoder_order_norm(model,order)
                if epoch==0:
                    checks.update(check_objective(model,output,batch,pairs))
            loss.backward()
            clips.append(torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True).item())
            optimizer.step()
            for key,value in {'loss':loss,'reconstruction':terms['reconstruction'],'kl_i':terms['kl_i'],
                              'kl_c':terms['kl_c'],'order':order,'l2':l2}.items():
                totals[key]+=value.detach().item()
        logs.append({'epoch':epoch+1,**{k:v/batches for k,v in totals.items()},'encoder_order_grad_norm':order_norm,
                     'mean_preclip_grad_norm':float(np.mean(clips))})
        pd.DataFrame(logs).to_csv(root/'losses.csv',index=False,float_format='%.17g')
        if epoch==0 or (epoch+1)%10==0 or epoch+1==epochs:
            print(f'epoch={epoch+1}/{epochs}: '+ ' '.join(f'{k}={logs[-1][k]:.6f}' for k in totals),flush=True)
    rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else torch.get_rng_state()
    rng_digest=hashlib.sha256(rng.numpy().tobytes()).hexdigest()
    if source_available and not smoke: assert rng_digest==old['reparameterization_rng_digest']
    model.eval()
    blocks={s:extract_blocks(model,dataset,16,device) for s,dataset in datasets.items()}
    with torch.no_grad(): raw={s:torch.sigmoid(-model.hi_head(values['I'][0])).cpu() for s,values in blocks.items()}
    scale={'min':raw['dev'].min().item(),'max':raw['dev'].max().item(),'count':len(raw['dev']),'fit_split':'dev'}
    saved={'stage':'ocald_no_kli','diagnostic_type':'zero_kl_i_s2_ablation','mode':settings['mode'],
           'epoch':epochs,'seed':42,'lambda':10.,'actual_training':actual,'config':config,
           'normalization':processed['normalization'],'columns':processed['columns'],
           'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
           'initial_state_dict':initial,'initial_sha256':state_hash(initial),'reparameterization_rng_digest':rng_digest,
           'raw_hi':raw,'hi_minmax':scale,
           'flight_keys':{s:[[f['unit'],f['cycle']] for f in d.flights] for s,d in datasets.items()}}
    torch.save(saved,root/'checkpoint.pt')
    dump_json(root/'mechanism_checks.json',checks)
    # First evaluation label access is after the last training update.
    evaluate(config,root/'checkpoint.pt')
    if source_available and not smoke:
        evaluate(config,source,root/'s2_reference')
    compare(root,source)
    checks.update(verify(root,device))
    dump_json(root/'verification.json', checks)
    notes=[f'no-KL_I {settings["mode"]}, seed42, lambda10, {epochs} joint epochs (not two-stage 100/100).',
           'Ablation objective=reconstruction+KL_C+10*order+1e-4*head_L2; KL_I logged only.',
           'No probes and no ranking metrics. Original S2 is evaluated read-only, never retrained.',
           'Initialization and sampling schedules checked against S2 where available; full noise RNG digest checked.',
           'HI/RUL only evaluation; dev-only raw scaling; no clipping, epsilon, direction flip or extra postprocessing.',
           'Single seed cannot establish a unique cause or physical disentanglement.',
           f'S2 comparison: {"re-evaluated" if source_available and not smoke else "not evaluated in smoke / source unavailable"}.']
    (root/'summary.txt').write_text('\n'.join(notes)+'\n')
    print(f'PASS: completed {settings["mode"]}; output={root}',flush=True)
    return root


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config.yaml')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--s2-checkpoint',default=DEFAULT_S2)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    if args.verify_only:
        if args.run_dir is None or args.smoke:parser.error('--verify-only requires --run-dir and excludes --smoke')
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type=='cpu':torch.set_num_threads(1)
        verify(args.run_dir,device)
    else:
        run(load_config(args.config),args.smoke,args.s2_checkpoint)


if __name__=='__main__':main()
