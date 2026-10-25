"""Two-condition coloring, keyed export reuse and inverse dev normalization."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

from dataset import NCMAPSSDataset
from experiments import visualize_continuous_conditions as continuous
from experiments import visualize_latent_flight_category as latent
from models.kgld import KGLD
from utils.preprocess import construct_u

ROOT = Path(__file__).resolve().parents[1]


class ContinuousConditions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_inverse_formula_and_channel_mapping_use_full_flight_u(self):
        # Deliberately permute raw channel order; select by W_var, never assumed index.
        names = ['TRA', 'Mach', 'T2', 'alt']
        raw = np.array([[60, .2, 490, 1000], [70, .7, 510, 9000], [80, .9, 520, 14000]], np.float32)
        mean = np.array([50, .4, 480, 6000.])
        std = np.array([10, .15, 20, 4000.])
        u = construct_u(raw, 2, 3, mean, std)[None]
        dataset = SimpleNamespace(samples=[(11, 3)], u=u,
                                  W_seq=torch.full((1, 100, 4), 999999.))
        result = continuous.flight_conditions(dataset, names, mean, std)
        self.assertEqual(result[['unit', 'cycle']].values.tolist(), [[11, 3]])
        self.assertAlmostEqual(result.mean_mach.iloc[0], float(raw[:, 1].mean()), places=6)
        self.assertAlmostEqual(result.mean_altitude.iloc[0], float(((raw[:, 3]-6000)/4000).mean()), places=6)
        recovered_alt = result.mean_altitude.iloc[0] * std[3] + mean[3]
        self.assertAlmostEqual(recovered_alt, float(raw[:, 3].mean()), places=3)
        with self.assertRaisesRegex(ValueError, 'Mach and alt'):
            continuous.flight_conditions(dataset, ['a', 'b', 'c', 'd'], mean, std)
        with self.assertRaisesRegex(ValueError, 'statistics'):
            continuous.flight_conditions(dataset, names, mean, np.zeros(4))

    def test_four_panels_share_samples_row_limits_and_column_color_norm(self):
        import matplotlib.pyplot as plt
        points = pd.DataFrame(dict(mu_d_0=[-1, 0, 1], mu_d_1=[-.5, 0, .5],
            PC1=[-.2, 0, .2], PC2=[.1, .3, .5], mean_mach=[.1, .5, .9], mean_altitude=[-2, 0, 3]))
        with tempfile.TemporaryDirectory() as directory, patch.object(plt, 'close'):
            ranges = continuous.plot_conditions(points, dict(explained_variance_ratio=[.4, .3]), Path(directory))
            figure = plt.gcf()
            panels = figure.axes[:4]
            self.assertEqual(len(figure.axes), 6)  # One shared colorbar per column.
            for index, ax in enumerate(panels):
                self.assertEqual(len(ax.collections[0].get_offsets()), 3)
                self.assertEqual(ax.get_aspect(), 1)
                condition = ['mean_mach', 'mean_altitude'][index % 2]
                self.assertEqual(ax.collections[0].norm.vmin, points[condition].min())
                self.assertEqual(ax.collections[0].norm.vmax, points[condition].max())
            for row in range(2):
                self.assertEqual(panels[row*2].get_xlim(), panels[row*2+1].get_xlim())
                self.assertEqual(panels[row*2].get_ylim(), panels[row*2+1].get_ylim())
                np.testing.assert_array_equal(panels[row*2].collections[0].get_offsets(),
                                              panels[row*2+1].collections[0].get_offsets())
            self.assertEqual(ranges['mean_altitude'], dict(vmin=-2., vmax=3.))
        plt.close(figure)

    def test_existing_exports_reused_without_network_or_pca_fit_and_reject_bad_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(63)
            raw = root/'raw.h5'
            with h5py.File(raw, 'w') as f:
                f['W_var'] = np.array(['alt', 'Mach', 'TRA', 'T2'], dtype='S4')
                for split, units in [('dev', [1, 2, 3]), ('test', [4, 5])]:
                    a = np.array([[u, c, 1, 1] for u in units for c in range(1, 5) for _ in range(8+c)])
                    f[f'A_{split}'] = a
                    f[f'W_{split}'] = rng.normal(size=(len(a), 4)) * [5000, .1, 15, 20] + [10000, .5, 60, 490]
                    f[f'X_s_{split}'] = rng.normal(size=(len(a), 14))
                    f[f'Y_{split}'] = (5-a[:, 1])[:, None]
            config = yaml.safe_load((ROOT/'config/kgld.yaml').read_text())
            config['data'].update(h5_path=str(raw), processed_dir=str(root/'processed'),
                                  downsample_factor=2, cycle_hi_path=None)
            config['model'].update(in_features=14, c_hidden=4, n_backbone_layers=1, n_decoder_layers=1)
            run = root/'run'; run.mkdir()
            config['run_dir'] = str(run)
            dev = NCMAPSSDataset(str(raw), config=config['data'], val_fraction=0)
            config['run_metadata'] = dict(processed_data=str(dev.processed_path))
            config_path = run/'config.yaml'; config_path.write_text(yaml.safe_dump(config))
            torch.manual_seed(63)
            torch.save(dict(model_state_dict=KGLD(config['model']).state_dict(),
                            run_metadata=config['run_metadata'], decoder_use_W_summary=True), run/'best.pt')
            exported = latent.run(str(config_path), str(run/'best.pt'), str(root/'latents'), 'cpu', 4)
            protected = [raw, run/'best.pt', *exported.iterdir(), dev.processed_path]
            before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
            with patch('sklearn.decomposition.PCA.fit', side_effect=AssertionError('must reuse PCA')), \
                    patch.object(latent, 'extract_condition_means', side_effect=AssertionError('must reuse means')):
                output = continuous.run(str(exported), str(root/'continuous'))
            points = pd.read_csv(output/'plot_data.csv')
            metadata = json.loads((output/'metadata.json').read_text())
            self.assertEqual(len(points), 8)
            self.assertEqual(metadata['units']['mean_altitude'], 'dev-global standardized mean (z-score)')
            self.assertTrue(metadata['pca_reused'])
            self.assertFalse(metadata['network_inference'])
            self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected})
            with self.assertRaises(FileExistsError):
                continuous.run(str(exported), str(output))
            pca_path = exported/'pca_parameters.json'
            saved = pca_path.read_text(); pca = json.loads(saved)
            pca['fit_keys'][0][0] = 'test'; pca_path.write_text(json.dumps(pca))
            with self.assertRaisesRegex(ValueError, 'exactly all dev'):
                continuous.load_plot_data(exported)
            pca_path.write_text(saved)
            projection_path = exported/'test_projection.csv'
            original = projection_path.read_text()
            pd.read_csv(projection_path).iloc[:-1].to_csv(projection_path,index=False)
            with self.assertRaisesRegex(ValueError, 'coverage mismatch'):
                continuous.load_plot_data(exported)
            projection_path.write_text(original)
            meta_path = exported/'metadata.json'; meta = json.loads(meta_path.read_text())
            meta['checkpoint']['sha256'] = 'incorrect'; meta_path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'checkpoint differs'):
                continuous.load_plot_data(exported)


if __name__ == '__main__':
    unittest.main()
