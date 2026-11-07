"""Keyed paper figure composition without model inference or PCA refitting."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import torch

from experiments.plot import plot_latent_summary as summary
from experiments.visualize_latent_flight_category import project_condition, D_COLUMNS, C_COLUMNS


def exports(root):
    health, operating = root/'health', root/'operating'
    health.mkdir(); operating.mkdir()
    rng = np.random.default_rng(8)
    features = pd.DataFrame(rng.normal(size=(18, 10)), columns=D_COLUMNS+C_COLUMNS)
    features['split'] = ['dev']*12+['test']*6
    features['unit'] = [1]*6+[2]*6+[3]*6
    features['cycle'] = list(range(1, 7))*3
    features['Fc'] = [1, 2, 3]*6
    projection, pca = project_condition(features)
    weight = np.array([.2, -.7], np.float32)
    checkpoint = root/'best.pt'
    torch.save(dict(model_state_dict={'hi_head.linear.weight': torch.from_numpy(weight[None])}), checkpoint)
    cache = root/'cache.h5'
    with h5py.File(cache,'w') as file:
        for split in ('dev','test'):
            file[f'{split}/keys'] = features.loc[features.split.eq(split), ['unit','cycle']].values
    metadata = dict(checkpoint=dict(path=str(checkpoint), sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()),
        data=dict(processed=str(cache), reference_key_provenance='synthetic explicit keys'),
        resolved_config=dict(model=dict(d_zd=2,d_zc=8),data=dict()),
        hi_head_weight=weight.astype(float).tolist(),category_encoding_basis='synthetic one-indexed Fc')
    (health/'projection.json').write_text(json.dumps(metadata))
    (operating/'metadata.json').write_text(json.dumps(metadata))
    points = projection.rename(columns=dict(mu_d_0='representation_0',mu_d_1='representation_1'))
    points['hi_gt'] = np.linspace(0,1,len(points))
    points['score'] = projection[D_COLUMNS].values @ weight.astype(float)
    points.iloc[::-1].to_csv(health/'latent_points.csv',index=False)
    projection.to_csv(operating/'test_projection.csv',index=False)
    features.to_csv(operating/'latent_features.csv',index=False)
    (operating/'pca_parameters.json').write_text(json.dumps(pca))
    return health, operating


class LatentSummary(unittest.TestCase):
    def test_reuse_key_join_and_reject_missing_keys_or_test_fitted_pca(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);health,operating=exports(root)
            sources=[*health.iterdir(),*operating.iterdir(),root/'best.pt',root/'cache.h5']
            before={p:p.read_bytes() for p in sources}
            with patch('sklearn.decomposition.PCA.fit',side_effect=AssertionError('must reuse PCA')):
                output=summary.run(str(health),str(operating),str(root/'output'))
            self.assertEqual(len(pd.read_csv(output/'plot_data.csv')),6)
            self.assertEqual(before,{p:p.read_bytes() for p in sources})
            with self.assertRaises(FileExistsError):
                summary.run(str(health),str(operating),str(output))
            path=operating/'test_projection.csv'; original=path.read_text()
            pd.read_csv(path).iloc[:-1].to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'coverage mismatch'):
                summary.load_points(health,operating)
            path.write_text(original)
            path=operating/'pca_parameters.json';pca=json.loads(path.read_text())
            pca['fit_keys'][0][0]='test';path.write_text(json.dumps(pca))
            with self.assertRaisesRegex(ValueError,'all dev'):
                summary.load_points(health,operating)

    def test_plot_uses_trained_direction_and_same_samples(self):
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);health,operating=exports(root)
            points,weight,metadata=summary.load_points(health,operating)
            with patch.object(plt,'close'):
                summary.plot_summary(points,weight,metadata,root)
                fig=plt.gcf();left,_,right=fig.axes
                self.assertEqual(len(left.collections[0].get_offsets()),len(points))
                self.assertEqual(sum(len(c.get_offsets()) for c in right.collections),len(points))
                arrow=left.texts[0]
                delta=np.asarray(arrow.xy)-np.asarray(arrow.xyann)
                np.testing.assert_allclose(delta/np.linalg.norm(delta),weight/np.linalg.norm(weight))
                self.assertEqual(left.get_title(),'(a) Degradation representation')
                self.assertEqual(right.get_title(),'(b) Operating-response representation')
                self.assertFalse(right.texts)
                self.assertTrue((root/'latent_summary.pdf').read_bytes().startswith(b'%PDF'))
            plt.close(fig)


if __name__=='__main__':
    unittest.main()
