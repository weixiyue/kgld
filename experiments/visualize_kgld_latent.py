"""Plot frozen two-dimensional KGLD degradation means and the trained HI direction."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset import NCMAPSSDataset
from evaluate import (_resolve_cycle_hi_path, _resolve_main_h5_path, associate,
                      load_evaluation_model, predict_cycles, read_evaluation_checkpoint,
                      read_reference, validate_keys)
from models import get_model_type
from utils.linear_probe import extract_representations

COORDINATES = ['representation_0', 'representation_1']
COLORS = ['#0072B2', '#D55E00', '#009E73']
FC_BASIS = ('utils/preprocess.py::construct_u: u=[W_mean(4), W_std(4), '
            'eye(n_fc)[Fc-1]]. No verified Short/Medium/Long mapping in this repository; '
            'use original one-indexed Fc identifiers.')
RTOL, ATOL = 2e-5, 2e-6


def flight_categories(dataset: NCMAPSSDataset, n_fc: int) -> pd.DataFrame:
    """Recover original Fc from u and attach the dataset's own flight keys."""
    u = dataset.u.detach().cpu().numpy()
    if n_fc != 3 or u.shape != (len(dataset.samples), 8 + n_fc):
        raise ValueError('expected u=[W_mean(4), W_std(4), Fc_onehot(3)]')
    onehot = u[:, -n_fc:]
    if (not np.isfinite(u).all() or not np.isin(onehot, [0, 1]).all()
            or not np.equal(onehot.sum(axis=1), 1).all()):
        raise ValueError('invalid finite u / Fc one-hot encoding')
    frame = pd.DataFrame(dataset.samples, columns=['unit', 'cycle'])
    validate_keys(frame, 'dataset Fc')
    return frame.assign(Fc=onehot.argmax(axis=1) + 1)


def align_points(representations: pd.DataFrame, labelled: pd.DataFrame,
                 categories: pd.DataFrame, weight: np.ndarray) -> tuple[pd.DataFrame, float]:
    """Join complete flight tables, then check the actual unscaled HI projection."""
    columns = [c for c in representations if c.startswith('representation_')]
    if columns != COORDINATES or weight.shape != (2,):
        raise ValueError('actual degradation representation and HIHead must have dimension 2; no reduction')
    if not np.isfinite(weight).all() or np.linalg.norm(weight) == 0:
        raise ValueError('HIHead weight must be finite and nonzero')
    frame = representations
    for name, other in [('public scores/reference', labelled), ('Fc', categories)]:
        validate_keys(frame, 'latent points')
        validate_keys(other, name)
        frame = frame.merge(other, on=['unit', 'cycle'], how='outer',
                            validate='one_to_one', indicator=True)
        if not frame['_merge'].eq('both').all():
            raise ValueError(f'latent/{name} flight-key coverage mismatch')
        frame = frame.drop(columns='_merge')
    frame = frame.assign(split='test')[['split', 'unit', 'cycle', *COORDINATES, 'Fc', 'hi_gt', 'score']]
    if not np.isfinite(frame.drop(columns='split').to_numpy(dtype=float)).all():
        raise ValueError('nonfinite latent points, scores or reference HI')
    projected = frame[COORDINATES].to_numpy() @ weight
    np.testing.assert_allclose(projected, frame.score, rtol=RTOL, atol=ATOL,
                               err_msg='mu_d @ trained HIHead differs from public raw score')
    return frame.sort_values(['split', 'unit', 'cycle']).reset_index(drop=True), float(
        np.max(np.abs(projected - frame.score.to_numpy())))


def compare_saved_scores(points: pd.DataFrame, checkpoint_file: Path, digest: str) -> list[dict]:
    """Compare existing evaluations only when metadata identifies the same checkpoint hash."""
    comparisons = []
    for path in sorted(checkpoint_file.parent.rglob('scores.csv')):
        metadata_path = path.with_name('metrics.json')
        if not metadata_path.is_file():
            comparisons.append(dict(path=str(path.resolve()), status='unverified checkpoint identity'))
            continue
        metadata = json.loads(metadata_path.read_text(encoding='utf-8')).get('metadata', {})
        if metadata.get('checkpoint', {}).get('sha256') != digest:
            comparisons.append(dict(path=str(path.resolve()), status='different/unverified checkpoint SHA256'))
            continue
        saved = pd.read_csv(path, float_precision='round_trip')
        saved = saved.loc[saved.split.eq('test'), ['unit', 'cycle', 'score']]
        validate_keys(saved, 'existing test scores')
        joined = points.merge(saved, on=['unit', 'cycle'], how='outer',
                              validate='one_to_one', indicator=True, suffixes=('', '_saved'))
        if not joined['_merge'].eq('both').all():
            raise ValueError(f'existing test score keys differ: {path}')
        np.testing.assert_allclose(joined.score, joined.score_saved, rtol=RTOL, atol=ATOL,
                                   err_msg=f'existing scores differ: {path}')
        comparisons.append(dict(path=str(path.resolve()), status='matched same checkpoint SHA256',
            n_test=len(saved), max_absolute_error=float(np.max(np.abs(joined.score - joined.score_saved)))))
    return comparisons


def plot_latent(points: pd.DataFrame, weight: np.ndarray, output: Path) -> dict:
    """Use unmodified latent coordinates, equal scales and the same translated arrow in both panels."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    xy = points[COORDINATES].to_numpy()
    low, high = xy.min(axis=0), xy.max(axis=0)
    span = float(np.max(high - low))
    if span == 0:
        span = 1.0  # Drawing extent only; coordinates remain unchanged.
    center = (low + high) / 2
    direction = weight / np.linalg.norm(weight)
    # Reserve a top band outside the full point cloud. Both arrow endpoints stay in it.
    length = 0.28 * span
    midpoint = np.array([center[0], high[1] + 0.27 * span])
    start, end = midpoint - direction * length / 2, midpoint + direction * length / 2
    xlim = [center[0] - 0.66 * span, center[0] + 0.66 * span]
    ylim = [low[1] - 0.12 * span, high[1] + 0.52 * span]
    with plt.rc_context({'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8), layout='constrained', sharex=True, sharey=True)
        scatter = axes[0].scatter(*xy.T, c=points.hi_gt, cmap='viridis', s=15, alpha=.75,
                                  edgecolors='none')
        fig.colorbar(scatter, ax=axes[0], label='Reference HI', fraction=.046, pad=.035)
        for fc, color in enumerate(COLORS, start=1):
            selected = points.Fc.eq(fc)
            axes[1].scatter(*xy[selected].T, color=color, s=15, alpha=.75,
                            edgecolors='none', label=f'Fc {fc}')
        axes[1].legend(loc='lower center', bbox_to_anchor=(.5, 1.02), ncol=3,
                       frameon=False, handletextpad=.3, columnspacing=.9)
        axes[0].set_title('(a) Reference health', pad=42)
        axes[1].set_title('(b) Flight category', pad=42)
        for ax in axes:
            ax.set(xlabel=r'$\mu_{d,1}$', ylabel=r'$\mu_{d,2}$', xlim=xlim, ylim=ylim)
            ax.set_aspect('equal', adjustable='box')
            ax.set_axisbelow(True)
            ax.grid(color='#e5e5e5', linewidth=.5)
            ax.annotate('', xy=end, xytext=start,
                        arrowprops=dict(arrowstyle='-|>', color='black', lw=1.6, mutation_scale=14))
            ax.text(midpoint[0], high[1] + .46 * span, r'$a$: increasing HI', ha='center', va='center')
        fig.savefig(output / 'latent_space.png', dpi=300, facecolor='white')
        fig.savefig(output / 'latent_space.pdf', facecolor='white')
        plt.close(fig)
    return dict(xlim=xlim, ylim=ylim, arrow_start=start.tolist(), arrow_end=end.tolist(),
                arrow='positive trained weight a; translated and uniformly scaled, not a time direction',
                coordinates='original posterior means; no per-coordinate scaling or dimensionality reduction')


def run(config_path: str, checkpoint_path: str, output_dir: str,
        device: str | None = None, batch_size: int | None = None, threads: int = 2) -> Path:
    """Extract all test flights using shared inference and reference association; fit nothing."""
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f'output directory already exists: {output}')
    if threads < 1:
        raise ValueError('threads must be positive')
    torch.set_num_threads(threads)
    config, flat, checkpoint, checkpoint_file, device = read_evaluation_checkpoint(
        config_path, checkpoint_path, device)
    if (get_model_type(config) != 'kgld' or config.get('experiment', {}).get('baseline_type')
            or config.get('data', {}).get('input_mode', 'flight') != 'flight'):
        raise ValueError('this experiment requires a KGLD flight-input checkpoint')
    if int(config['model'].get('d_zd', 2)) != 2:
        raise ValueError('degradation representation must have dimension 2; no automatic reduction')
    batch_size = batch_size if batch_size is not None else config.get('evaluation', {}).get('batch_size', 64)
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    digest = hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()
    h5_path = _resolve_main_h5_path(config['data'])
    reference_path = _resolve_cycle_hi_path(flat, h5_path, 'test')
    if reference_path is None:
        raise ValueError('this visualization requires test reference HI')
    protected = {Path(p).resolve() for p in (h5_path, reference_path, config_path, checkpoint_file)}
    dataset = NCMAPSSDataset(h5_path, None, flat, split='test')
    protected.add(dataset.processed_path.resolve())
    recorded_cache = checkpoint.get('processed_data', checkpoint.get('run_metadata', {}).get('processed_data'))
    if recorded_cache is not None and Path(recorded_cache).resolve() != dataset.processed_path.resolve():
        raise ValueError('prepared data differs from checkpoint preprocessing source')
    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in protected}
    model = load_evaluation_model(config, flat, checkpoint, dataset)
    model.requires_grad_(False).to(device).eval()
    if model.hi_head.linear.bias is not None:
        raise ValueError('expected the trained bias-free HIHead')
    weight = model.hi_head.linear.weight.detach().cpu().numpy().astype(float).reshape(-1)
    with torch.no_grad():
        representations = extract_representations(model, dataset, device, batch_size)
        scores = predict_cycles(model, dataset, device, batch_size)
    reference, provenance = read_reference(reference_path, dataset.rul)
    labelled = associate(scores, dataset.rul, reference)
    points, error = align_points(representations, labelled,
                                 flight_categories(dataset, int(flat['n_fc'])), weight)
    comparisons = compare_saved_scores(points, checkpoint_file, digest)
    after = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in protected}
    if before != after or hashlib.sha256(checkpoint_file.read_bytes()).hexdigest() != digest:
        raise RuntimeError('checkpoint or input source changed during inference')
    output.mkdir(parents=True, exist_ok=False)
    plotting = plot_latent(points, weight, output)
    points.to_csv(output / 'latent_points.csv', index=False)
    metadata = dict(
        experiment='frozen KGLD two-dimensional degradation space and trained HI projection',
        checkpoint=dict(path=str(checkpoint_file.resolve()), sha256=digest, epoch=checkpoint.get('epoch')),
        config_path=str(Path(config_path).resolve()), resolved_config=config,
        training_run_metadata=checkpoint.get('run_metadata', {}),
        data=dict(raw=str(Path(h5_path).resolve()), processed=str(dataset.processed_path.resolve()),
                  reference=str(Path(reference_path).resolve()), reference_key_provenance=provenance),
        split='test', n_flights=len(points), units=sorted(points.unit.unique().astype(int).tolist()),
        category_counts={str(int(k)): int(v) for k, v in points.Fc.value_counts().sort_index().items()},
        category_encoding_basis=FC_BASIS, category_labels={str(i): f'Fc {i}' for i in (1, 2, 3)},
        coordinates=dict(representation_0='mu_d posterior mean dimension 1',
                         representation_1='mu_d posterior mean dimension 2'),
        hi_head_weight=weight.tolist(), score='raw public predict_cycles score = mu_d @ a, no scaling',
        public_final_HI='not recomputed; public dev-global Min-Max is a positive affine map of score',
        max_projection_absolute_error=error, comparison_tolerance=dict(rtol=RTOL, atol=ATOL),
        existing_score_comparisons=comparisons, device=device, batch_size=batch_size, threads=threads,
        torch_version=str(torch.__version__), plotting=plotting,
        safeguards=dict(checkpoint_sha256_unchanged=True, input_size_and_mtime_unchanged=True,
                        frozen_eval=True, gradients=False, sampling=False, fitted_probe=False),
    )
    (output / 'projection.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    correlations = points[[*COORDINATES, 'hi_gt', 'score']].corr(method='spearman').hi_gt
    note = (
        f'# KGLD latent space\n\nCheckpoint: `{checkpoint_file.resolve()}` (epoch {checkpoint.get("epoch")}).\n\n'
        f'All {len(points)} test flights; units {metadata["units"]}. Fc counts: {metadata["category_counts"]}. '
        f'No selection by HI, category or model score. Maximum |mu_d @ a - score|: {error:.9g}.\n\n'
        f'Descriptive pooled Spearman correlation with reference HI: mu_d,1={correlations.iloc[0]:.6g}, '
        f'mu_d,2={correlations.iloc[1]:.6g}, raw score={correlations.loc["score"]:.6g}. '
        'These are diagnostics, not fitted directions or evidence of causal disentanglement.\n\n'
        'The arrow uses only the trained HIHead, points toward increasing HI and is not a degradation-time arrow. '
        'Both plots show identical unmodified points and limits. Category overlap alone cannot demonstrate complete '
        'disentanglement. Reference HI is used only for coloring and diagnostics. No training, PCA or supervised probe.\n\n'
        f'Fc labels: {FC_BASIS}\n\nReference key limitation: {provenance}.\n'
    )
    (output / 'README.md').write_text(note, encoding='utf-8')
    print(f'Saved {len(points)} test flights to {output}; max projection error={error:.9g}', flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='confirmed paper run/config.yaml')
    parser.add_argument('--checkpoint', required=True, help='explicit confirmed trained checkpoint')
    parser.add_argument('--output-dir', required=True, help='new independent directory')
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    run(args.config, args.checkpoint, args.output_dir, args.device, args.batch_size, args.threads)


if __name__ == '__main__':
    main()
