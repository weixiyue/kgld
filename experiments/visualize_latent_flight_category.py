"""Compare raw degradation means with dev-fitted PCA of operating-response means."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
import torch
from torch.utils.data import DataLoader

from dataset import NCMAPSSDataset
from evaluate import (_resolve_main_h5_path, load_evaluation_model,
                      read_evaluation_checkpoint, validate_keys)
from experiments.visualize_kgld_latent import COLORS, FC_BASIS, RTOL, ATOL, flight_categories
from models import get_model_type
from utils.linear_probe import extract_representations

KEYS = ['split', 'unit', 'cycle']
D_COLUMNS = [f'mu_d_{i}' for i in range(2)]
C_COLUMNS = [f'mu_c_{i}' for i in range(8)]


def extract_condition_means(model: torch.nn.Module, dataset: NCMAPSSDataset,
                            device: str, batch_size: int) -> pd.DataFrame:
    """Use the original backbone and FiLM branch to extract complete mu_c, without decoding."""
    model.requires_grad_(False).to(device).eval()
    keys, vectors = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        for x, mask, u, _, units, cycles, _ in loader:
            x, mask, u = x.to(device), mask.to(device), u.to(device)
            mean = model.zc_branch(model.backbone(x, mask), mask, u)[0]
            if mean.shape != (len(x), 8) or not torch.isfinite(mean).all():
                raise ValueError('expected finite eight-dimensional mu_c posterior means')
            keys.extend(zip(units.tolist(), cycles.tolist(), strict=True))
            vectors.append(mean.cpu().numpy())
    frame = pd.DataFrame(keys, columns=['unit', 'cycle'])
    frame[C_COLUMNS] = np.concatenate(vectors).astype(float)
    validate_keys(frame, 'mu_c')
    return frame


def join_features(degradation: pd.DataFrame, condition: pd.DataFrame,
                  categories: pd.DataFrame, split: str) -> pd.DataFrame:
    """Attach explicit split keys before one-to-one joins; reject dropped or repeated flights."""
    if [c for c in degradation if c.startswith('representation_')] != ['representation_0', 'representation_1']:
        raise ValueError('expected complete two-dimensional mu_d, without reduction')
    frame = degradation.rename(columns={f'representation_{i}': D_COLUMNS[i] for i in range(2)})
    frame = frame.assign(split=split)
    for name, other in [('mu_c', condition), ('Fc', categories)]:
        validate_keys(frame, f'{split} features')
        validate_keys(other, name)
        frame = frame.merge(other.assign(split=split), on=KEYS, how='outer',
                            validate='one_to_one', indicator=True)
        if not frame['_merge'].eq('both').all():
            raise ValueError(f'{split}: feature/{name} key coverage mismatch')
        frame = frame.drop(columns='_merge')
    return frame[KEYS + D_COLUMNS + C_COLUMNS + ['Fc']].sort_values(KEYS).reset_index(drop=True)


def project_condition(features: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fit unscaled ordinary PCA on ALL dev means, then transform test means only."""
    if features.duplicated(KEYS).any() or not set(features.split) == {'dev', 'test'}:
        raise ValueError('expected unique dev/test split/unit/cycle keys')
    for split, group in features.groupby('split'):
        validate_keys(group, split)
    if not np.isfinite(features[D_COLUMNS + C_COLUMNS + ['Fc']].to_numpy()).all():
        raise ValueError('nonfinite feature or category')
    if not features.Fc.isin([1, 2, 3]).all():
        raise ValueError('invalid Fc identifier')
    dev = features.loc[features.split.eq('dev')].sort_values(KEYS)
    test = features.loc[features.split.eq('test')].sort_values(KEYS).copy()
    pca = PCA(n_components=2, svd_solver='full', whiten=False)
    pca.fit(dev[C_COLUMNS].to_numpy())
    if not np.isfinite(pca.explained_variance_ratio_).all():
        raise ValueError('PCA explained variance is undefined for these dev means')
    test[['PC1', 'PC2']] = pca.transform(test[C_COLUMNS].to_numpy())
    if not np.isfinite(test[['PC1', 'PC2']].to_numpy()).all():
        raise ValueError('nonfinite PCA test coordinates')
    parameters = dict(n_components=2, svd_solver='full', whiten=False, standardization=False,
        feature_columns=C_COLUMNS, mean=pca.mean_.tolist(), components=pca.components_.tolist(),
        explained_variance=pca.explained_variance_.tolist(),
        explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
        cumulative_explained_variance_ratio=float(pca.explained_variance_ratio_.sum()),
        singular_values=pca.singular_values_.tolist(), fit_split='dev', n_samples_seen=int(pca.n_samples_),
        fit_units=sorted(dev.unit.unique().astype(int).tolist()),
        fit_keys=dev[KEYS].values.tolist(), fit_labels_used=False, test_operation='transform only')
    return test[KEYS + D_COLUMNS + ['PC1', 'PC2', 'Fc']].reset_index(drop=True), parameters


def compare_previous_degradation(test: pd.DataFrame, checkpoint_file: Path, digest: str) -> list[dict]:
    """Find previous keyed latent plots by checkpoint hash, without relying on stale run paths."""
    comparisons = []
    for path in sorted(checkpoint_file.resolve().parent.parent.rglob('projection.json')):
        points_path = path.with_name('latent_points.csv')
        if not points_path.is_file():
            continue
        metadata = json.loads(path.read_text(encoding='utf-8'))
        if metadata.get('checkpoint', {}).get('sha256') != digest:
            continue
        previous = pd.read_csv(points_path, float_precision='round_trip')
        previous = previous.loc[previous.split.eq('test'), KEYS + ['representation_0', 'representation_1']]
        validate_keys(previous, 'previous latent means')
        joined = test.merge(previous, on=KEYS, how='outer', validate='one_to_one', indicator=True)
        if not joined['_merge'].eq('both').all():
            raise ValueError(f'previous latent flight-key coverage mismatch: {points_path}')
        old = joined[['representation_0', 'representation_1']].to_numpy()
        new = joined[D_COLUMNS].to_numpy()
        np.testing.assert_allclose(new, old, rtol=RTOL, atol=ATOL,
                                   err_msg=f'mu_d differs from previous latent experiment: {points_path}')
        comparisons.append(dict(path=str(points_path.resolve()), checkpoint_sha256_matched=True,
            n_test=len(joined), max_absolute_error=float(np.max(np.abs(new - old)))))
    return comparisons


def plot_comparison(test: pd.DataFrame, parameters: dict, output: Path) -> None:
    """Draw all test flights with independent equal-aspect latent and PCA coordinate axes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ratio = parameters['explained_variance_ratio']
    with plt.rc_context({'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11,
                         'axes.titlesize': 11, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
        for ax, columns in zip(axes, [D_COLUMNS, ['PC1', 'PC2']], strict=True):
            for fc, color in enumerate(COLORS, start=1):
                xy = test.loc[test.Fc.eq(fc), columns].to_numpy()
                ax.scatter(*xy.T, color=color, label=f'Fc {fc}', s=13, alpha=.65, edgecolors='none')
            ax.set_aspect('equal', adjustable='box')
            ax.set_axisbelow(True)
            ax.grid(color='#e5e5e5', linewidth=.5)
            ax.margins(.10)
            ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=3, frameon=False)
        axes[0].set(xlabel=r'$\mu_{d,1}$', ylabel=r'$\mu_{d,2}$')
        axes[1].set(xlabel=f'PC1 ({ratio[0]:.2%})', ylabel=f'PC2 ({ratio[1]:.2%})')
        axes[0].set_title('(a) Degradation representation', pad=38)
        axes[1].set_title('(b) Operating-response representation', pad=38)
        fig.savefig(output / 'latent_flight_category.png', dpi=300, facecolor='white')
        fig.savefig(output / 'latent_flight_category.pdf', facecolor='white')
        plt.close(fig)


def run(config_path: str, checkpoint_path: str, output_dir: str, device: str | None = None,
        batch_size: int = 32, threads: int = 2) -> Path:
    """Extract frozen flight means, fit dev PCA and save a new independent experiment."""
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f'output directory already exists: {output}')
    if batch_size < 1 or threads < 1:
        raise ValueError('batch_size and threads must be positive')
    torch.set_num_threads(threads)
    config, flat, checkpoint, checkpoint_file, device = read_evaluation_checkpoint(
        config_path, checkpoint_path, device)
    if (get_model_type(config) != 'kgld' or config.get('experiment', {}).get('baseline_type')
            or config['data'].get('input_mode', 'flight') != 'flight'):
        raise ValueError('expected a KGLD flight-input checkpoint')
    if (config['model'].get('d_zd', 2), config['model'].get('d_zc', 8)) != (2, 8):
        raise ValueError('expected actual mu_d dimension 2 and mu_c dimension 8; no slicing')
    digest = hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()
    h5_path = _resolve_main_h5_path(config['data'])
    datasets = dict(dev=NCMAPSSDataset(h5_path, None, flat, val_fraction=0),
                    test=NCMAPSSDataset(h5_path, None, flat, split='test'))
    cache = datasets['dev'].processed_path.resolve()
    recorded_cache = checkpoint.get('processed_data', checkpoint.get('run_metadata', {}).get('processed_data'))
    if recorded_cache is not None and Path(recorded_cache).resolve() != cache:
        raise ValueError('prepared data differs from checkpoint preprocessing source')
    model = load_evaluation_model(config, flat, checkpoint, datasets['dev'])
    model.requires_grad_(False).to(device).eval()
    frames = []
    with torch.no_grad():
        for split, dataset in datasets.items():
            frames.append(join_features(extract_representations(model, dataset, device, batch_size),
                extract_condition_means(model, dataset, device, batch_size),
                flight_categories(dataset, int(flat['n_fc'])), split))
    features = pd.concat(frames, ignore_index=True)
    test, parameters = project_condition(features)
    comparisons = compare_previous_degradation(test, checkpoint_file, digest)
    if hashlib.sha256(checkpoint_file.read_bytes()).hexdigest() != digest:
        raise RuntimeError('checkpoint changed during inference')
    counts = {split: dict(n_flights=len(frame), units=sorted(frame.unit.unique().astype(int).tolist()),
                Fc_counts={str(int(k)): int(v) for k, v in frame.Fc.value_counts().sort_index().items()})
              for split, frame in features.groupby('split')}
    metadata = dict(checkpoint=dict(path=str(checkpoint_file.resolve()), sha256=digest,
                    epoch=checkpoint.get('epoch')), config_path=str(Path(config_path).resolve()),
        resolved_config=config, training_run_metadata=checkpoint.get('run_metadata', {}),
        data=dict(raw=str(Path(h5_path).resolve()), processed=str(cache)),
        device=device, batch_size=batch_size, threads=threads,
        torch_version=str(torch.__version__), sklearn_version=sklearn.__version__,
        category_encoding_basis=FC_BASIS, category_colors={str(i): c for i, c in enumerate(COLORS, 1)},
        samples=counts, feature_columns=dict(mu_d=D_COLUMNS, mu_c=C_COLUMNS),
        coordinate_indexing='CSV latent indices are zero-based; plotted mu_d indices are one-based',
        previous_mu_d_comparisons=comparisons,
        previous_mu_d_check='passed' if comparisons else 'unavailable: no matching prior latent_points.csv found',
        frozen_eval=True, no_grad=True, sampling=False,
        pca_protocol='raw mu_c; all dev flights fit, test transform only; no scaling or labels',
        interpretation='mu_c FiLM receives u including Fc: class structure can show encoded flight-category '
                       'information, not complete disentanglement or out-of-sample category recognition')
    output.mkdir(parents=True, exist_ok=False)
    features.to_csv(output / 'latent_features.csv', index=False)
    test.to_csv(output / 'test_projection.csv', index=False)
    (output / 'pca_parameters.json').write_text(json.dumps(parameters, indent=2, allow_nan=False), encoding='utf-8')
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    plot_comparison(test, parameters, output)
    ratio = parameters['explained_variance_ratio']
    note = (f'# Flight-category representation comparison\n\nCheckpoint: {checkpoint_file.resolve()} '
        f'(epoch {checkpoint.get("epoch")}, SHA256 {digest}).\n\n'
        f'Samples: {json.dumps(counts)}. All test flights appear in both panels.\n\n'
        f'PCA: {len(datasets["dev"])} dev flights, original eight-dimensional mu_c, no standardization '
        f'or labels, full SVD; test transform only. PC1={ratio[0]:.4%}, PC2={ratio[1]:.4%}, '
        f'cumulative={sum(ratio):.4%}. Remaining variance is not shown; overlap in two dimensions '
        'does not establish absence of class information in eight dimensions.\n\n'
        f'Prior mu_d comparison: {metadata["previous_mu_d_check"]}; {json.dumps(comparisons)}.\n\n'
        f'Fc encoding: {FC_BASIS}\n\n'
        'The operating-response branch receives u including Fc through FiLM. Class separation may support '
        'encoded flight-category information, but alone cannot prove complete disentanglement or '
        'generalization of flight-category recognition. No network training, labels for PCA, HI projection '
        'arrow or supervised probe is used.\n')
    (output / 'README.md').write_text(note, encoding='utf-8')
    print(f'Saved {counts}; PCA variance ratios={ratio}, cumulative={sum(ratio):.6f}; output={output}', flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    run(args.config, args.checkpoint, args.output_dir, args.device, args.batch_size, args.threads)


if __name__ == '__main__':
    main()
