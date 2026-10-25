"""Frozen posterior means, keyed joins and dev-only PCA without network training."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch

from dataset import NCMAPSSDataset
from evaluate import read_evaluation_checkpoint, load_evaluation_model
from experiments import visualize_latent_flight_category as visual
from test_linear_probe_comparison import fixtures
from utils.linear_probe import extract_representations


class FlightCategoryLatents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_condition_means_equal_eval_forward_and_do_not_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            source = fixtures(Path(directory))[0]
            config, flat, cp, _, _ = read_evaluation_checkpoint(str(source / 'config.yaml'),
                                                               str(source / 'best.pt'), 'cpu')
            dataset = NCMAPSSDataset(config['data']['h5_path'], config=flat, split='test')
            model = load_evaluation_model(config, flat, cp, dataset)
            rng = torch.get_rng_state().clone()
            actual = visual.extract_condition_means(model, dataset, 'cpu', 4)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertFalse(model.training)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
            with torch.no_grad():
                expected = model(dataset.x, dataset.mask, dataset.u, dataset.W_seq)['mu_c'].numpy()
            np.testing.assert_allclose(actual[visual.C_COLUMNS], expected, rtol=2e-5, atol=2e-6)
            another_batch = visual.extract_condition_means(model, dataset, 'cpu', 1)
            np.testing.assert_allclose(actual[visual.C_COLUMNS], another_batch[visual.C_COLUMNS],
                                       rtol=2e-5, atol=2e-6)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, cp['model_state_dict'][name], rtol=0, atol=0)
            model.zc_branch.d_zc = 7
            with self.assertRaisesRegex(ValueError, 'eight-dimensional'):
                visual.extract_condition_means(model, dataset, 'cpu', 4)

    def test_keyed_join_rejects_missing_duplicates_and_wrong_dimension(self):
        d = pd.DataFrame(dict(unit=[2, 1], cycle=[1, 2], representation_0=[1., 2.],
                               representation_1=[3., 4.]))
        c = d[['unit', 'cycle']].copy()
        c[visual.C_COLUMNS] = np.arange(16).reshape(2, 8)
        fc = d[['unit', 'cycle']].assign(Fc=[3, 1])
        joined = visual.join_features(d, c.iloc[::-1], fc.iloc[::-1], 'test')
        self.assertEqual(joined.Fc.tolist(), [1, 3])
        self.assertEqual(joined.mu_c_0.tolist(), [8, 0])
        self.assertTrue(joined.split.eq('test').all())
        with self.assertRaisesRegex(ValueError, 'coverage mismatch'):
            visual.join_features(d, c.iloc[:1], fc, 'test')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            visual.join_features(d, pd.concat([c, c.iloc[:1]]), fc, 'test')
        with self.assertRaisesRegex(ValueError, 'two-dimensional'):
            visual.join_features(d.assign(representation_2=0), c, fc, 'test')

    def test_pca_fits_all_dev_raw_means_and_only_transforms_test(self):
        rng = np.random.default_rng(61)
        features = pd.DataFrame(rng.normal(size=(18, 10)), columns=visual.D_COLUMNS + visual.C_COLUMNS)
        features['split'] = ['dev'] * 12 + ['test'] * 6
        features['unit'] = [1] * 6 + [2] * 6 + [3] * 6
        features['cycle'] = list(range(1, 7)) * 3
        features['Fc'] = [1, 2, 3] * 6
        test, parameters = visual.project_condition(features)
        oracle = PCA(n_components=2, svd_solver='full').fit(features.iloc[:12][visual.C_COLUMNS])
        np.testing.assert_array_equal(parameters['mean'], oracle.mean_)
        np.testing.assert_array_equal(parameters['components'], oracle.components_)
        np.testing.assert_allclose(test[['PC1', 'PC2']], oracle.transform(features.iloc[12:][visual.C_COLUMNS]))
        np.testing.assert_array_equal(test[visual.D_COLUMNS], features.iloc[12:][visual.D_COLUMNS])
        restored = (features.iloc[12:][visual.C_COLUMNS].to_numpy() - parameters['mean']) @ np.array(parameters['components']).T
        np.testing.assert_allclose(restored, test[['PC1', 'PC2']], atol=1e-14)
        changed = features.copy()
        changed.loc[changed.split.eq('test'), visual.C_COLUMNS] += 10000
        changed['Fc'] = 3
        _, unchanged = visual.project_condition(changed)
        self.assertEqual(parameters, unchanged)
        self.assertEqual(parameters['n_samples_seen'], 12)
        self.assertTrue(all(key[0] == 'dev' for key in parameters['fit_keys']))
        for invalid in [pd.concat([features, features.iloc[:1]]),
                        features.assign(mu_c_0=np.nan), features.assign(Fc=4)]:
            with self.assertRaises(ValueError):
                visual.project_condition(invalid)

    def test_end_to_end_and_previous_mean_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixtures(root)[0]
            config, flat, cp, checkpoint, _ = read_evaluation_checkpoint(str(source / 'config.yaml'),
                                                                       str(source / 'best.pt'), 'cpu')
            test = NCMAPSSDataset(config['data']['h5_path'], config=flat, split='test')
            model = load_evaluation_model(config, flat, cp, test)
            previous = source.parent / 'previous_plot'
            previous.mkdir()
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            (previous / 'projection.json').write_text(json.dumps(dict(checkpoint=dict(sha256=digest))))
            extract_representations(model, test, 'cpu', 4).assign(split='test').iloc[::-1].to_csv(
                previous / 'latent_points.csv', index=False)
            before = {p: p.read_bytes() for p in [checkpoint, source / 'config.yaml', *previous.iterdir()]}
            output = visual.run(str(source / 'config.yaml'), str(checkpoint), str(root / 'output'), 'cpu', 4)
            metadata = json.loads((output / 'metadata.json').read_text())
            parameters = json.loads((output / 'pca_parameters.json').read_text())
            features = pd.read_csv(output / 'latent_features.csv')
            projection = pd.read_csv(output / 'test_projection.csv')
            self.assertEqual(len(features), 36)
            self.assertEqual(len(projection), 12)
            self.assertFalse(features.duplicated(visual.KEYS).any())
            self.assertEqual(metadata['previous_mu_d_check'], 'passed')
            self.assertEqual(metadata['previous_mu_d_comparisons'][0]['max_absolute_error'], 0)
            np.testing.assert_allclose(parameters['mean'], features.loc[features.split.eq('dev'), visual.C_COLUMNS].mean())
            self.assertEqual(metadata['samples']['test']['n_flights'], 12)
            self.assertTrue((output / 'latent_flight_category.pdf').read_bytes().startswith(b'%PDF'))
            self.assertTrue((output / 'latent_flight_category.png').stat().st_size > 1000)
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            with self.assertRaises(FileExistsError):
                visual.run(str(source / 'config.yaml'), str(checkpoint), str(output), 'cpu', 4)
            changed = projection.copy()
            changed['mu_d_0'] += 100
            with self.assertRaises(AssertionError):
                visual.compare_previous_degradation(changed, checkpoint, digest)
            self.assertEqual(visual.compare_previous_degradation(projection, checkpoint, 'another hash'), [])


if __name__ == '__main__':
    unittest.main()
