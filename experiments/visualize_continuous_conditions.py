"""Color existing KGLD latent coordinates by flight mean Mach and altitude only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from dataset import NCMAPSSDataset
from evaluate import _resolve_main_h5_path, read_evaluation_checkpoint, validate_keys
from experiments.visualize_latent_flight_category import KEYS, D_COLUMNS, C_COLUMNS


def flight_conditions(dataset: NCMAPSSDataset, names: list[str], mean: np.ndarray,
                      std: np.ndarray) -> pd.DataFrame:
    """Read pre-truncation means from u; invert dev normalization only for dimensionless Mach."""
    if len(names) != 4 or len(set(names)) != 4 or not {'Mach', 'alt'} <= set(names):
        raise ValueError('W_var must identify unique Mach and alt channels in the four input channels')
    if mean.shape != (4,) or std.shape != (4,) or not np.isfinite([mean, std]).all() or (std <= 0).any():
        raise ValueError('invalid cached dev-global W statistics')
    normalized = dataset.u.detach().cpu().numpy()[:, :4]
    if normalized.shape != (len(dataset.samples), 4) or not np.isfinite(normalized).all():
        raise ValueError('invalid cached flight W means in u')
    # construct_u casts both raw W and global statistics to float32 before normalization.
    recovered = normalized.astype(float) * std.astype(np.float32) + mean.astype(np.float32)
    frame = pd.DataFrame(dataset.samples, columns=['unit', 'cycle']).assign(split='test')
    validate_keys(frame, 'flight conditions')
    return frame.assign(mean_mach=recovered[:, names.index('Mach')],
                        mean_altitude=normalized[:, names.index('alt')].astype(float))


def load_plot_data(latent_dir: Path) -> tuple[pd.DataFrame, dict, dict]:
    """Validate provenance and reuse existing dev PCA parameters and complete latent exports."""
    metadata_path = latent_dir / 'metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    checkpoint_path = Path(metadata['checkpoint']['path'])
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if digest != metadata['checkpoint']['sha256']:
        raise ValueError('checkpoint differs from the existing latent experiment')
    config, flat, checkpoint, _, _ = read_evaluation_checkpoint(
        metadata['config_path'], str(checkpoint_path), 'cpu')
    if config != metadata['resolved_config']:
        raise ValueError('resolved config differs from exported latent experiment')
    raw = _resolve_main_h5_path(config['data'])
    if Path(raw).resolve() != Path(metadata['data']['raw']).resolve():
        raise ValueError('raw-data source differs from the existing latent experiment')
    dataset = NCMAPSSDataset(raw, None, flat, split='test')
    cache = dataset.processed_path.resolve()
    recorded = checkpoint.get('processed_data', checkpoint.get('run_metadata', {}).get('processed_data'))
    if cache != Path(metadata['data']['processed']).resolve() or (recorded and cache != Path(recorded).resolve()):
        raise ValueError('preprocessing cache differs from the latent experiment or checkpoint')
    with h5py.File(raw, 'r') as file:
        names = [v.decode() if isinstance(v, bytes) else str(v) for v in file['W_var'][()].reshape(-1)]
    with h5py.File(cache, 'r') as file:
        mean, std = file['W_mean_global'][()], file['W_std_global'][()]
        dev_keys = pd.DataFrame(file['dev/keys'][()], columns=['unit', 'cycle']).assign(split='dev')
        test_keys = pd.DataFrame(file['test/keys'][()], columns=['unit', 'cycle']).assign(split='test')
    features_path = latent_dir / 'latent_features.csv'
    features = pd.read_csv(features_path, float_precision='round_trip')
    if set(features.split) != {'dev', 'test'} or features.duplicated(KEYS).any():
        raise ValueError('latent features must have unique dev/test flight keys')
    if not np.isfinite(features[D_COLUMNS + C_COLUMNS].to_numpy()).all():
        raise ValueError('nonfinite exported latent features')
    for split, keys in [('dev', dev_keys), ('test', test_keys)]:
        group = features.loc[features.split.eq(split)]
        validate_keys(group, f'{split} latent features')
        if group[KEYS].sort_values(KEYS).values.tolist() != keys[KEYS].sort_values(KEYS).values.tolist():
            raise ValueError(f'{split} latent/cache flight-key coverage mismatch')
    pca_path = latent_dir / 'pca_parameters.json'
    pca = json.loads(pca_path.read_text(encoding='utf-8'))
    if (pca['fit_split'] != 'dev' or pca['svd_solver'] != 'full' or pca['n_components'] != 2
            or pca['standardization'] or pca['whiten'] or pca['feature_columns'] != C_COLUMNS
            or pca['n_samples_seen'] != len(dev_keys)
            or pca['fit_keys'] != dev_keys[KEYS].sort_values(KEYS).values.tolist()):
        raise ValueError('PCA must fit exactly all dev raw mu_c means, with full SVD and no scaling')
    components, center = np.asarray(pca['components']), np.asarray(pca['mean'])
    if components.shape != (2, 8) or center.shape != (8,) or not np.isfinite(components).all() or not np.isfinite(center).all():
        raise ValueError('invalid saved PCA numerical parameters')
    dev = features.loc[features.split.eq('dev')]
    np.testing.assert_allclose(center, dev[C_COLUMNS].mean().to_numpy(), rtol=1e-12, atol=1e-14)
    projection_path = latent_dir / 'test_projection.csv'
    projection = pd.read_csv(projection_path, float_precision='round_trip')
    if not projection.split.eq('test').all():
        raise ValueError('plot projection must contain test flights only')
    validate_keys(projection, 'existing test projection')
    test = features.loc[features.split.eq('test')]
    joined = test.merge(projection[KEYS + D_COLUMNS + ['PC1', 'PC2']], on=KEYS,
                        how='outer', validate='one_to_one', indicator=True, suffixes=('', '_plot'))
    if not joined['_merge'].eq('both').all():
        raise ValueError('feature/projection flight-key coverage mismatch')
    np.testing.assert_array_equal(joined[D_COLUMNS], joined[[c + '_plot' for c in D_COLUMNS]])
    np.testing.assert_allclose((joined[C_COLUMNS].to_numpy() - center) @ components.T,
                               joined[['PC1', 'PC2']], rtol=1e-12, atol=1e-14)
    conditions = flight_conditions(dataset, names, mean, std)
    points = projection[KEYS + D_COLUMNS + ['PC1', 'PC2']].merge(
        conditions, on=KEYS, how='outer', validate='one_to_one', indicator=True)
    if not points['_merge'].eq('both').all():
        raise ValueError('condition/projection flight-key coverage mismatch')
    points = points.drop(columns='_merge').sort_values(KEYS).reset_index(drop=True)
    if not np.isfinite(points.drop(columns='split').to_numpy()).all():
        raise ValueError('nonfinite plot coordinates or conditions')
    provenance = dict(checkpoint=metadata['checkpoint'], resolved_config=config,
        data=metadata['data'], latent_source=str(latent_dir.resolve()),
        sources={p.name: dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                 for p in [metadata_path, features_path, pca_path, projection_path]},
        pca_reused=True, network_inference=False, pca_fit_split='dev', n_pca_fit=len(dev_keys),
        explained_variance_ratio=pca['explained_variance_ratio'],
        W_channel_order=names, selected_channels=dict(mach=names.index('Mach'), altitude=names.index('alt')),
        W_statistics=dict(source=str(cache), fit_split='dev', mean=mean.tolist(), std=std.tolist(),
                          operation='compute_W_stats: all downsampled dev timesteps; std + 1e-8'),
        condition_calculation='construct_u: float32 raw W and float32 dev statistics; '
            'mean of normalized full downsampled flight BEFORE truncation/padding. '
            'Mach raw_mean = u[channel] * float32(W_std_global[channel]) + float32(W_mean_global[channel]); '
            'altitude uses u[channel] because source altitude units are not documented.',
        units=dict(mean_mach='dimensionless', mean_altitude='dev-global standardized mean (z-score)'),
        altitude_unit_basis='raw HDF W_var identifies alt, but no physical-unit attributes or verified '
            'unit mapping in repository; deliberately do not label meters or feet',
        n_test=len(points), test_units=sorted(points.unit.unique().astype(int).tolist()))
    return points, provenance, pca


def plot_conditions(points: pd.DataFrame, pca: dict, output: Path) -> dict:
    """Use one full-test color scale per column and identical latent coordinates within each row."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    columns = ['mean_mach', 'mean_altitude']
    ranges = {c: dict(vmin=float(points[c].min()), vmax=float(points[c].max())) for c in columns}
    ratio = pca['explained_variance_ratio']
    with plt.rc_context({'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 10,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white', 'pdf.fonttype': 42}):
        fig = plt.figure(figsize=(10, 8), layout='constrained')
        grid = fig.add_gridspec(2, 4, width_ratios=[1, .045, 1, .045])
        axes = np.empty((2, 2), dtype=object)
        for row in range(2):
            for col in range(2):
                axes[row, col] = fig.add_subplot(grid[row, 2 * col],
                    sharex=axes[row, 0] if col else None, sharey=axes[row, 0] if col else None)
        for col, condition in enumerate(columns):
            norm = Normalize(**ranges[condition])
            for row, xy_columns in enumerate([D_COLUMNS, ['PC1', 'PC2']]):
                ax = axes[row, col]
                scatter = ax.scatter(*points[xy_columns].to_numpy().T, c=points[condition],
                    cmap='viridis', norm=norm, s=13, alpha=.75, edgecolors='none')
                ax.set_aspect('equal', adjustable='box')
                ax.set_axisbelow(True)
                ax.grid(color='#e5e5e5', linewidth=.5)
                ax.margins(.10)
                ax.set_xlabel(r'$\mu_{d,1}$' if row == 0 else f'PC1 ({ratio[0]:.2%})')
                ax.set_ylabel(r'$\mu_{d,2}$' if row == 0 else f'PC2 ({ratio[1]:.2%})')
            fig.colorbar(scatter, cax=fig.add_subplot(grid[:, 2 * col + 1]),
                         label='Mean Mach number' if col == 0 else 'Mean altitude (dev z-score)')
        axes[0, 0].set_title('Flight mean Mach number', pad=12)
        axes[0, 1].set_title('Flight mean altitude', pad=12)
        fig.savefig(output / 'continuous_condition_latent.png', dpi=300, facecolor='white')
        fig.savefig(output / 'continuous_condition_latent.pdf', facecolor='white')
        plt.close(fig)
    return ranges


def run(latent_dir: str, output_dir: str) -> Path:
    """Supplement a validated exported experiment without network inference or refitting PCA."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f'output directory already exists: {output}')
    points, metadata, pca = load_plot_data(Path(latent_dir))
    output.mkdir(parents=True, exist_ok=False)
    metadata['color_ranges'] = plot_conditions(points, pca, output)
    metadata['all_panels_same_test_keys'] = True
    points.to_csv(output / 'plot_data.csv', index=False)
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    (output / 'README.md').write_text(
        '# Continuous-condition latent comparison\n\n'
        f'Reuses {Path(latent_dir).resolve()} and the same paper checkpoint. '
        f'All {len(points)} test flights; no network inference, PCA refit, training or supervised readout.\n\n'
        'Mach is recovered from cached u with dev W statistics. Altitude is explicitly shown in '
        'dev z-score units because source physical units are not verified. Neither uses padded W_seq.\n\n'
        f'PC1/PC2 variance ratios: {pca["explained_variance_ratio"]}. '
        'The saved PCA was fit on all dev raw mu_c means; test was transformed only.\n\n'
        'Color gradients describe associations with these two conditions, not complete disentanglement.\n',
        encoding='utf-8')
    print(f'Saved {len(points)} test flights to {output}; color ranges={metadata["color_ranges"]}', flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--latent-dir', default='runs/kgld_latent_flight_category',
                        help='existing full posterior exports, dev PCA and provenance; prepare with visualize_latent_flight_category if absent')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    run(args.latent_dir, args.output_dir)


if __name__ == '__main__':
    main()
