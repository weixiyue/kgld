"""Keyed DS03 evaluation and comparison using one dev-only score HI mapping."""

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import FlightDataset, load_processed, read_reference, read_rul
from model import fit_score_scale, scale_score, validate_score_scale
from train import extract_training_means, load_ocald


def attach_references(predictions, hi, rul):
    keys = ["unit", "cycle"]
    for name, table in (("predictions", predictions), ("HI", hi), ("RUL", rul)):
        if table.duplicated(keys).any():
            raise ValueError(f"Duplicate {name} unit/cycle keys")
    expected = set(map(tuple, predictions[keys].to_numpy()))
    if set(map(tuple, hi[keys].to_numpy())) != expected or set(map(tuple, rul[keys].to_numpy())) != expected:
        raise ValueError("Missing or unexpected HI/RUL associations")
    result = predictions.merge(hi, on=keys, validate="one_to_one").merge(rul, on=keys, validate="one_to_one")
    if result[["score", "hi_gt", "rul"]].isna().any().any():
        raise ValueError("Missing score or evaluation reference values")
    return result.sort_values(keys).reset_index(drop=True)


def predict_table(config=None, checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(1)
    model, saved = load_ocald(checkpoint_path, device)
    config = saved['config']  # The checkpoint snapshot is authoritative.
    processed = load_processed(config)
    if saved["columns"] != processed["columns"]:
        raise ValueError("Checkpoint columns differ from prepared data")
    for field in ("x", "w"):
        for key in ("min", "max"):
            if not torch.equal(saved["normalization"][field][key], processed["normalization"][field][key]):
                raise ValueError("Checkpoint normalization differs from full dev normalization")
    tables = []
    for split in ("dev", "test"):
        dataset = FlightDataset(processed, split)
        means, units, cycles = extract_training_means(model, dataset, config["training"]["representation"]["batch_size"], device)
        with torch.no_grad():
            score = model.score(means).cpu()
        if not torch.isfinite(score).all():
            raise ValueError(f"Nonfinite score in {split}")
        if "scores" in saved and split in saved["scores"]:
            torch.testing.assert_close(score, saved["scores"][split], rtol=1e-6, atol=1e-7)
        if split == "dev":
            fitted = fit_score_scale(score)
            if "hi_mapping" in saved:
                mapping = saved["hi_mapping"]
                validate_score_scale(mapping)
                for key in ('type', 'fit_split', 'count', 'available'):
                    if mapping[key] != fitted[key]:
                        raise ValueError("Saved mapping is not fitted on the complete dev split")
                np.testing.assert_allclose([mapping['min'], mapping['max']], [fitted['min'], fitted['max']], rtol=1e-6, atol=1e-7)
                source = "checkpoint_hi_mapping"
            else:
                # Old hi_minmax and score_minmax are deliberately never read.
                mapping = fitted
                source = "all_dev_final_scores_from_loaded_model"
        hi_values = scale_score(score, mapping).numpy() if mapping['available'] else np.full(len(score), np.nan)
        predictions = pd.DataFrame({"split": split, "unit": units.cpu().numpy(), "cycle": cycles.cpu().numpy(),
                                    "score": score.numpy(), "hi": hi_values})
        # Labels are loaded only after score fitting and mapping.
        hi = read_reference(config["data"][f"hi_{split}_path"], processed[split])
        rul = read_rul(config["data"]["raw_path"], split, processed[split])
        tables.append(attach_references(predictions, hi, rul))
    checkpoint = {**saved, "hi_mapping": mapping, "mapping_source": source}
    return pd.concat(tables, ignore_index=True), checkpoint


def metric_report(table, checkpoint):
    from metrics import evaluate_metrics

    result = {"stage": "evaluation", "training_stage": checkpoint.get("stage"), "mode": checkpoint["mode"],
              "epoch": checkpoint["epoch"], "source_epoch": checkpoint.get("source_epoch"),
              "seed": checkpoint["config"]["training"]["seed"], "structure": checkpoint["config"]["model"],
              "loss_weights": checkpoint.get("loss_weights"), "hi_mapping": checkpoint["hi_mapping"],
              "mapping_source": checkpoint.get("mapping_source", "checkpoint_hi_mapping"),
              "reference_commit": "157ebfae7baeee1e0cd52180f8e58ead3dcb448e", "splits": {}}
    for split in ("dev", "test"):
        subset = table.loc[table.split == split]
        if not result['hi_mapping']['available']:
            if not subset.hi.isna().all():
                raise ValueError("Zero score range must mark all HI values unavailable")
            result['splits'][split] = {"hi": {"available": False, "reason": "Dev score range is zero; HI unavailable"}}
            continue
        if subset.hi.isna().any():
            raise ValueError("Missing HI with an available dev score mapping")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            values = evaluate_metrics(subset.hi.to_numpy(), subset.hi_gt.to_numpy(),
                                      subset.unit.to_numpy(), subset.cycle.to_numpy(), subset.rul.to_numpy(), result['seed'])
        for entries in [values['overall'], *values['per_unit'].values()]:
            for entry in entries.values():
                if not np.isfinite(entry['value']):
                    original = entry.get('reason')
                    entry['value'] = None
                    entry['reason'] = "Original metric formula returned a nonfinite value; no epsilon or clipping" + (f"; {original}" if original else "")
        result['splits'][split] = {"hi": {"available": True, **values, "warnings": sorted({str(w.message) for w in caught})}}
    return result


def plot_curves(table, report, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = list(table.groupby(["split", "unit"], sort=True))
    figure, axes = plt.subplots(len(groups), 1, figsize=(10, 2.5 * len(groups)), layout="constrained", squeeze=False)
    try:
        for row, ((split, unit), group) in enumerate(groups):
            group = group.sort_values('cycle')
            axis = axes[row, 0]
            axis.plot(group.cycle, group.hi_gt, color='black', linewidth=1, label='Reference HI')
            entry = report['splits'][split]['hi']
            if entry['available']:
                line, = axis.plot(group.cycle, group.hi, linewidth=1.1, label='Predicted HI (dev score scale)')
                if not np.array_equal(line.get_xdata(), group.cycle.to_numpy()) or not np.array_equal(line.get_ydata(), group.hi.to_numpy()):
                    raise ValueError('Curve coordinates differ from predictions')
                metrics = entry['per_unit'][str(int(unit))]
                details = ' '.join(f"{key}={metrics[key]['value']:.3f}" if metrics[key]['value'] is not None else f"{key}=N/A"
                                   for key in ('mae', 'mape', 'monotonicity', 'spearman', 'mi'))
            else:
                details = entry['reason']
            axis.set_title(f'{split} unit {int(unit)} | {details}', fontsize=9)
            axis.set_xlabel('Flight cycle'); axis.set_ylabel('HI'); axis.grid(alpha=.25); axis.legend(fontsize=8)
        figure.savefig(output_path, dpi=140)
    finally:
        plt.close(figure)


def evaluate(config, checkpoint_path, output_dir=None):
    run_dir = Path(checkpoint_path).parent if output_dir is None else Path(output_dir)
    if any((run_dir / name).exists() for name in ('predictions.csv', 'metrics.json', 'hi_curves.png', 'score_scaling.json')):
        raise ValueError('Refusing to overwrite evaluation artifacts; supply a new output directory')
    table, checkpoint = predict_table(config, checkpoint_path)
    run_dir.mkdir(parents=True, exist_ok=output_dir is None)
    metadata = {'hi_mapping': checkpoint['hi_mapping'], 'mapping_source': checkpoint['mapping_source'],
                'source_checkpoint': str(checkpoint_path), 'checkpoint_sha256': hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()}
    (run_dir/'score_scaling.json').write_text(json.dumps(metadata, indent=2)+'\n')
    table.to_csv(run_dir/'predictions.csv', index=False, float_format='%.17g')
    exported = pd.read_csv(run_dir/'predictions.csv', float_precision='round_trip')
    report = metric_report(exported, checkpoint)
    plot_curves(exported, report, run_dir/'hi_curves.png')
    (run_dir/'metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(f"Evaluation mode={checkpoint['mode']}: {len(table)} rows; output={run_dir}")
    for split, entries in report['splits'].items():
        print(split, entries['hi'].get('overall', entries['hi']))
    return run_dir


def compare(evaluations, output_dir):
    """Compare exported model evaluations; never fit parameters or select models."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root = Path(output_dir)
    names = ('original', 'joint', 'joint_w')
    if set(evaluations) != set(names):
        raise ValueError('Comparison requires original, joint and joint_w evaluation directories')
    if any((root/f).exists() for f in ('comparison.csv', 'comparison_dev_curves.png', 'comparison_test_curves.png', 'comparison_sources.json')):
        raise ValueError('Refusing to overwrite comparison products')
    tables, reports, sources, rows = {}, {}, {}, []
    for name in names:
        directory = Path(evaluations[name])
        tables[name] = pd.read_csv(directory/'predictions.csv', float_precision='round_trip')
        reports[name] = json.loads((directory/'metrics.json').read_text())
        sources[name] = json.loads((directory/'score_scaling.json').read_text())
        if reports[name]['seed'] != 42 or reports[name]['mode'] not in ('formal', 'full'):
            raise ValueError('Comparison requires seed42 formal models')
        validate_score_scale(reports[name]['hi_mapping'])
        if reports[name]['hi_mapping']['count'] != int((tables[name].split == 'dev').sum()):
            raise ValueError('Comparison mapping must cover all dev flights')
        model = reports[name]['structure']
        if model['latent_i_dim'] != 8 or model['latent_c_dim'] != 8 or model.get('score_mode', 'hi_head') != 'hi_head':
            raise ValueError('Comparison requires the three HIHead models with 8D I/C')
        if model.get('decoder_use_w', False) != (name == 'joint_w'):
            raise ValueError('Comparison model decoder W path mismatch')
        if reports[name]['epoch'] != 100 or reports[name]['training_stage'] != ('hi' if name == 'original' else 'joint'):
            raise ValueError('Comparison requires full last-epoch models')
        if name == 'original' and reports[name]['source_epoch'] != 100:
            raise ValueError('Original model requires 100/100 epochs')
        expected_weights = dict(reconstruction=0., kl_i=0., kl_c=0., order=1., l2=.0001) if name == 'original' else dict(reconstruction=1., kl_i=1., kl_c=1., order=10., l2=.0001)
        if reports[name]['loss_weights'] != expected_weights:
            raise ValueError('Comparison loss weights differ from the three approved versions')
        keys = ['split', 'unit', 'cycle']
        if not tables[name][keys].equals(tables['original'][keys]):
            raise ValueError('Comparison flight keys differ')
        for field in ('hi_gt', 'rul'):
            np.testing.assert_array_equal(tables[name][field], tables['original'][field])
        for split in ('dev', 'test'):
            entry = reports[name]['splits'][split]['hi']
            rows.append({'model': name, 'split': split, 'available': entry['available'],
                         **{k: v['value'] for k, v in entry.get('overall', {}).items()}})
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root/'comparison.csv', index=False, float_format='%.17g')
    (root/'comparison_sources.json').write_text(json.dumps(sources, indent=2)+'\n')
    for split in ('dev', 'test'):
        groups = list(tables['original'].loc[tables['original'].split == split].groupby('unit'))
        figure, axes = plt.subplots((len(groups)+2)//3, 3, figsize=(18, 3.5*((len(groups)+2)//3)), layout='constrained', squeeze=False)
        try:
            for axis, (unit, reference) in zip(axes.flat, groups):
                axis.plot(reference.cycle, reference.hi_gt, color='black', label='Reference HI', linewidth=1)
                for name in names:
                    group = tables[name].loc[(tables[name].split == split) & (tables[name].unit == unit)]
                    if reports[name]['splits'][split]['hi']['available']:
                        line, = axis.plot(group.cycle, group.hi, label=name, linewidth=1)
                        np.testing.assert_array_equal(line.get_ydata(), group.hi.to_numpy())
                axis.set_title(f'{split} unit {unit}'); axis.set_xlabel('Flight cycle'); axis.set_ylabel('HI'); axis.grid(alpha=.25); axis.legend(fontsize=8)
            for axis in list(axes.flat)[len(groups):]:
                axis.set_visible(False)
            figure.savefig(root/f'comparison_{split}_curves.png', dpi=140)
        finally:
            plt.close(figure)
    print(pd.DataFrame(rows).to_string(index=False))
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='Legacy argument; use checkpoint configuration snapshot')
    parser.add_argument('--checkpoint')
    parser.add_argument('--output-dir')
    parser.add_argument('--compare', nargs=3, metavar='NAME=EVALUATION_DIR')
    args = parser.parse_args()
    if args.compare:
        if args.checkpoint or not args.output_dir:
            parser.error('--compare requires --output-dir and cannot accompany --checkpoint')
        compare(dict(item.split('=', 1) for item in args.compare), args.output_dir)
    elif args.checkpoint:
        evaluate(None, args.checkpoint, args.output_dir)
    else:
        parser.error('Supply --checkpoint or --compare')


if __name__ == '__main__':
    main()
