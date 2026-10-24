"""Keyed latent plots and synthetic checkpoint inference; no network training."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from dataset import NCMAPSSDataset
from evaluate import load_evaluation_model, predict_cycles, read_evaluation_checkpoint
from experiments import visualize_kgld_latent as visual
from test_linear_probe_comparison import fixtures


class LatentVisualization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_fc_from_u_and_keys_not_unit_number(self):
        dataset = SimpleNamespace(samples=[(15, 2), (10, 1), (15, 1)],
            u=torch.cat([torch.zeros(3, 8), torch.eye(3)[[2, 0, 1]]], dim=1))
        frame = visual.flight_categories(dataset, 3)
        self.assertEqual(frame.Fc.tolist(), [3, 1, 2])
        self.assertEqual(frame[['unit', 'cycle']].values.tolist(), [[15, 2], [10, 1], [15, 1]])
        for onehot in ([[0, 0, 0]], [[1, 1, 0]], [[.5, .5, 0]], [[float('nan'), 0, 1]]):
            dataset = SimpleNamespace(samples=[(1, 1)], u=torch.tensor([[0.] * 8 + onehot[0]]))
            with self.assertRaisesRegex(ValueError, 'one-hot'):
                visual.flight_categories(dataset, 3)

    def test_keyed_alignment_dimension_and_projection(self):
        reps = pd.DataFrame(dict(unit=[2, 1, 1], cycle=[1, 2, 1],
                                 representation_0=[3., 2., 1.], representation_1=[6., 3., -2.]))
        labels = reps[['unit', 'cycle']].assign(score=[0., 1., 4.], hi_gt=[.1, .5, .9])
        fc = reps[['unit', 'cycle']].assign(Fc=[3, 2, 1])
        weight = np.array([2., -1.])
        points, error = visual.align_points(reps, labels.iloc[::-1], fc.iloc[[1, 0, 2]], weight)
        self.assertEqual(points.Fc.tolist(), [1, 2, 3])
        self.assertEqual(error, 0)
        for table, message in [(reps.assign(representation_2=0), 'dimension 2'),
                               (pd.concat([reps, reps.iloc[:1]]), 'duplicate'),
                               (reps.iloc[:2], 'coverage'),
                               (reps.assign(representation_0=np.nan), 'nonfinite')]:
            with self.assertRaisesRegex(ValueError, message):
                visual.align_points(table, labels, fc, weight)
        with self.assertRaisesRegex(ValueError, 'nonzero'):
            visual.align_points(reps, labels, fc, np.zeros(2))
        with self.assertRaises(AssertionError):
            visual.align_points(reps, labels.assign(score=99), fc, weight)

    def test_plot_original_coordinates_limits_and_direction(self):
        points = pd.DataFrame(dict(representation_0=[-2., 0, 1], representation_1=[-.3, .1, .4],
                                   Fc=[1, 2, 3], hi_gt=[.9, .5, .1]))
        weight = np.array([2., -3.])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            metadata = visual.plot_latent(points, weight, output)
            delta = np.array(metadata['arrow_end']) - metadata['arrow_start']
            np.testing.assert_allclose(delta / np.linalg.norm(delta), weight / np.linalg.norm(weight))
            self.assertTrue((output / 'latent_space.png').stat().st_size > 1000)
            self.assertTrue((output / 'latent_space.pdf').read_bytes().startswith(b'%PDF'))
            for index, coordinate in enumerate(visual.COORDINATES):
                limits = metadata['xlim' if index == 0 else 'ylim']
                self.assertLess(limits[0], points[coordinate].min())
                self.assertGreater(limits[1], points[coordinate].max())

    def test_integration_frozen_source_unchanged_and_existing_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixtures(root)[0]
            cp_path = source / 'best.pt'
            cp_before = cp_path.read_bytes()
            config, flat, cp, _, _ = read_evaluation_checkpoint(str(source / 'config.yaml'), str(cp_path), 'cpu')
            dataset = NCMAPSSDataset(config['data']['h5_path'], config=flat, split='test')
            model = load_evaluation_model(config, flat, cp, dataset)
            scores = predict_cycles(model, dataset, 'cpu', 3).assign(split='test')
            scores.iloc[::-1].to_csv(source / 'scores.csv', index=False)
            (source / 'metrics.json').write_text(json.dumps(dict(metadata=dict(
                checkpoint=dict(sha256=hashlib.sha256(cp_before).hexdigest())))))
            old_files = {p: p.read_bytes() for p in source.iterdir() if p.is_file()}
            captured = []
            original = visual.extract_representations

            def extract(model, *args):
                captured.append(model)
                self.assertFalse(model.training)
                self.assertFalse(torch.is_grad_enabled())
                self.assertTrue(all(not p.requires_grad for p in model.parameters()))
                return original(model, *args)

            with patch.object(visual, 'extract_representations', side_effect=extract):
                output = visual.run(str(source / 'config.yaml'), str(cp_path), str(root / 'plots'), 'cpu', 4)
            metadata = json.loads((output / 'projection.json').read_text())
            points = pd.read_csv(output / 'latent_points.csv')
            self.assertEqual(len(points), 12)
            self.assertEqual(metadata['units'], [5, 6])
            self.assertEqual(metadata['existing_score_comparisons'][0]['n_test'], 12)
            self.assertTrue(metadata['existing_score_comparisons'][0]['scores_within_tolerance'])
            self.assertLess(metadata['max_projection_absolute_error'], 1e-6)
            self.assertEqual(metadata['resolved_config'], config)
            self.assertEqual(set(p.name for p in output.iterdir()),
                {'latent_points.csv', 'projection.json', 'latent_space.png', 'latent_space.pdf', 'README.md'})
            for key, value in captured[0].state_dict().items():
                torch.testing.assert_close(value.cpu(), cp['model_state_dict'][key], rtol=0, atol=0)
            self.assertTrue(all(p.grad is None for p in captured[0].parameters()))
            self.assertEqual(old_files, {p: p.read_bytes() for p in old_files})
            with self.assertRaises(FileExistsError):
                visual.run(str(source / 'config.yaml'), str(cp_path), str(output), 'cpu')
            scores.assign(score=scores.score + 1).to_csv(source / 'scores.csv', index=False)
            discrepancy = visual.compare_saved_scores(points, cp_path, metadata['checkpoint']['sha256'])[0]
            self.assertFalse(discrepancy['scores_within_tolerance'])
            self.assertEqual(discrepancy['status'], 'score difference exceeds tolerance')
            # Matching checkpoint identity must not hide mismatching flight keys.
            scores.iloc[:-1].to_csv(source / 'scores.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'keys differ'):
                visual.compare_saved_scores(points, cp_path, metadata['checkpoint']['sha256'])


if __name__ == '__main__':
    unittest.main()
