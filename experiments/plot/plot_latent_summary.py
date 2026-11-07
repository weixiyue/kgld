"""Combine existing paper-checkpoint health and flight-category latent plots."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from evaluate import validate_keys
from experiments.visualize_kgld_latent import COLORS, RTOL, ATOL
from experiments.visualize_latent_flight_category import KEYS, D_COLUMNS, C_COLUMNS


def load_points(health_dir: Path, operating_dir: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Join verified test exports and reuse the trained HI weight and dev PCA without fitting."""
    health = json.loads((health_dir / 'projection.json').read_text())
    operating = json.loads((operating_dir / 'metadata.json').read_text())
    checkpoint = Path(health['checkpoint']['path'])
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if any(m['checkpoint']['sha256'] != digest for m in (health, operating)):
        raise ValueError('latent exports must use the same actual checkpoint SHA256')
    if (Path(health['data']['processed']).resolve() != Path(operating['data']['processed']).resolve()
            or health['resolved_config']['model'] != operating['resolved_config']['model']
            or health['resolved_config']['data'] != operating['resolved_config']['data']):
        raise ValueError('latent exports differ in model or preprocessing source')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    weight = saved['model_state_dict']['hi_head.linear.weight'].detach().numpy().astype(float).reshape(-1)
    if weight.shape != (2,) or not np.isfinite(weight).all() or np.linalg.norm(weight) == 0:
        raise ValueError('expected a finite nonzero two-dimensional HIHead weight')
    np.testing.assert_array_equal(weight, health['hi_head_weight'])
    left = pd.read_csv(health_dir / 'latent_points.csv', float_precision='round_trip')
    right = pd.read_csv(operating_dir / 'test_projection.csv', float_precision='round_trip')
    for name, frame in [('health', left), ('operating', right)]:
        validate_keys(frame, name)
        if not frame.split.eq('test').all():
            raise ValueError(f'{name} export must contain test flights only')
    joined = right.merge(left, on=KEYS, how='outer', validate='one_to_one', indicator=True,
                          suffixes=('', '_health'))
    if not joined['_merge'].eq('both').all():
        raise ValueError('health/operating test flight-key coverage mismatch')
    np.testing.assert_allclose(joined[D_COLUMNS], joined[['representation_0', 'representation_1']],
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(joined.Fc, joined.Fc_health)
    points = joined[KEYS + D_COLUMNS + ['PC1', 'PC2', 'Fc', 'hi_gt', 'score']].sort_values(KEYS).reset_index(drop=True)
    if not np.isfinite(points.drop(columns='split').to_numpy()).all() or not points.Fc.isin([1, 2, 3]).all():
        raise ValueError('nonfinite coordinates/HI or invalid flight category')
    np.testing.assert_allclose(points[D_COLUMNS].to_numpy() @ weight, points.score, rtol=RTOL, atol=ATOL)
    features = pd.read_csv(operating_dir / 'latent_features.csv', float_precision='round_trip')
    if features.duplicated(KEYS).any() or set(features.split) != {'dev', 'test'}:
        raise ValueError('expected unique dev/test latent feature keys')
    with h5py.File(operating['data']['processed'], 'r') as cache:
        for split in ('dev', 'test'):
            expected = cache[f'{split}/keys'][()].tolist()
            actual = features.loc[features.split.eq(split)].sort_values(['unit', 'cycle'])
            validate_keys(actual, split)
            if actual[['unit', 'cycle']].values.tolist() != sorted(expected):
                raise ValueError(f'{split} feature/cache key mismatch')
    pca = json.loads((operating_dir / 'pca_parameters.json').read_text())
    dev = features.loc[features.split.eq('dev')].sort_values(KEYS)
    if (pca['fit_keys'] != dev[KEYS].values.tolist() or pca['fit_split'] != 'dev'
            or pca['svd_solver'] != 'full' or pca['standardization'] or pca['whiten']):
        raise ValueError('PCA must use all dev raw posterior means without scaling')
    test = features.loc[features.split.eq('test')]
    checked = points.merge(test[KEYS + C_COLUMNS], on=KEYS, how='outer', validate='one_to_one', indicator=True)
    if not checked['_merge'].eq('both').all():
        raise ValueError('plot/full-feature test key mismatch')
    np.testing.assert_allclose((checked[C_COLUMNS].to_numpy() - pca['mean']) @ np.array(pca['components']).T,
                               checked[['PC1', 'PC2']], rtol=1e-12, atol=1e-14)
    metadata = dict(checkpoint=health['checkpoint'], data=health['data'],
        health_source=str(health_dir.resolve()), operating_source=str(operating_dir.resolve()),
        hi_head_weight=weight.tolist(), pca_source=str((operating_dir / 'pca_parameters.json').resolve()),
        explained_variance_ratio=pca['explained_variance_ratio'], n_pca_fit=len(dev), pca_fit_split='dev',
        n_test=len(points), units=sorted(points.unit.unique().astype(int).tolist()),
        Fc_counts={str(int(k)): int(v) for k, v in points.Fc.value_counts().sort_index().items()},
        identical_test_keys=True, network_inference=False, pca_refit=False,
        category_basis=operating['category_encoding_basis'],
        reference_key_provenance=health['data']['reference_key_provenance'])
    return points, weight, metadata


def plot_summary(points: pd.DataFrame, weight: np.ndarray, metadata: dict, output: Path) -> None:
    """Plot both branches for the same flights; only the left panel has the trained HI arrow."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    xy = points[D_COLUMNS].to_numpy()
    low, high = xy.min(axis=0), xy.max(axis=0)
    span = float(np.max(high - low)) or 1.
    center = (low + high) / 2
    middle = np.array([center[0], high[1] + .27 * span])
    delta = weight / np.linalg.norm(weight) * .28 * span
    ratio = metadata['explained_variance_ratio']
    with plt.rc_context({'font.family': 'serif', 'font.size': 10, 'axes.titlesize': 11,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white', 'pdf.fonttype': 42}):
        fig = plt.figure(figsize=(10.5, 4.8), layout='constrained')
        grid = fig.add_gridspec(1, 3, width_ratios=[1, .045, 1.15])
        left, bar, right = [fig.add_subplot(grid[0, i]) for i in range(3)]
        scatter = left.scatter(*xy.T, c=points.hi_gt, cmap='viridis', s=14, alpha=.75, edgecolors='none')
        fig.colorbar(scatter, cax=bar, label='Reference HI')
        left.set(xlabel=r'$\mu_{d,1}$', ylabel=r'$\mu_{d,2}$',
                 xlim=(center[0]-.66*span, center[0]+.66*span),
                 ylim=(low[1]-.12*span, high[1]+.52*span))
        left.annotate('', xy=middle+delta/2, xytext=middle-delta/2,
                      arrowprops=dict(arrowstyle='-|>', color='black', lw=1.6, mutation_scale=14))
        left.text(middle[0], high[1]+.46*span, 'Increasing HI', ha='center', va='center')
        for fc, color in enumerate(COLORS, 1):
            selected = points.Fc.eq(fc)
            right.scatter(points.loc[selected, 'PC1'], points.loc[selected, 'PC2'],
                          color=color, label=f'Fc {fc}', s=14, alpha=.65, edgecolors='none')
        right.set(xlabel=f'PC1 ({ratio[0]:.2%})', ylabel=f'PC2 ({ratio[1]:.2%})')
        right.legend(loc='upper right', title='Flight category', frameon=False)
        right.margins(.1)
        for ax in (left, right):
            ax.set_aspect('equal', adjustable='box')
            ax.set_axisbelow(True)
            ax.grid(color='#e5e5e5', linewidth=.5)
        left.set_title('(a) Degradation representation', pad=12)
        right.set_title('(b) Operating-response representation', pad=12)
        fig.savefig(output / 'latent_summary.png', dpi=300, facecolor='white')
        fig.savefig(output / 'latent_summary.pdf', facecolor='white')
        plt.close(fig)


def run(health_dir: str, operating_dir: str, output_dir: str) -> Path:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f'output directory already exists: {output}')
    points, weight, metadata = load_points(Path(health_dir), Path(operating_dir))
    output.mkdir(parents=True, exist_ok=False)
    plot_summary(points, weight, metadata, output)
    points.to_csv(output / 'plot_data.csv', index=False)
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Combined plot saved to {output}; identical test flights={len(points)}', flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--health-dir', default='runs/kgld_latent_visualization')
    parser.add_argument('--operating-dir', default='runs/kgld_latent_flight_category')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    run(args.health_dir, args.operating_dir, args.output_dir)


if __name__ == '__main__':
    main()
