"""CPU-only replay: recovered GT inputs must reproduce the saved 200k model."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import yaml

from GRALE.data.dataset import custom_collate_fn
from GRALE.loss.geometry import decode_graph, summarize_groups
from GRALE.main import GRALE_model
from scripts.run_spatial_anchor_pilot import forward, inspection_datasets
from scripts.inspect_geometric_run import digest, geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--inspection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve existing verification outputs; select a new path')
    torch.set_num_threads(1)
    reference = json.loads((args.reference_run / 'result.json').read_text())
    manifest = Path(reference['subset'])
    assert digest(manifest) == reference['subset_sha256']
    frozen = json.loads(manifest.read_text())
    config = yaml.safe_load((args.reference_run / 'config.yaml').read_text())
    datasets, provenance = inspection_datasets(manifest, frozen, config,
                                               args.inspection, args.reference_run)
    checkpoint = args.reference_run / 'last.ckpt'
    checkpoint_sha = digest(checkpoint)
    report = json.loads((args.inspection / 'report.json').read_text())
    assert checkpoint_sha == report['checkpoint_sha256'] == reference['checkpoint_sha256']
    model = GRALE_model.load_from_checkpoint(str(checkpoint), map_location='cpu').eval()
    records, groups = [], {}
    with torch.no_grad():
        for split, dataset in datasets.items():
            scores = []
            for index, sample in enumerate(dataset.samples):
                graph = dataset[index]
                output, _, _, _ = forward(model, custom_collate_fn([graph]))
                prediction = decode_graph(model.format_logits(output.clone())[0])
                with np.load(args.reference_run / split / f'{sample["sample_id"]}.npz',
                             allow_pickle=False) as saved:
                    same = (np.array_equal(prediction['slots'], saved['slots'])
                            and np.array_equal(prediction['edges'], saved['edges'])
                            and prediction['nodes'].shape == saved['nodes'].shape
                            and np.allclose(prediction['nodes'], saved['nodes'], atol=1e-5, rtol=1e-5))
                records.append({'split': split, 'sample_id': sample['sample_id'],
                                'reproduces_saved_prediction': bool(same)})
                scores.append(geometry(prediction, decode_graph(graph), dataset.load_target(index)['metric']))
            groups[split] = summarize_groups(scores, dataset.samples)
    result = {'reference': str(args.reference_run), 'checkpoint_sha256': checkpoint_sha,
              'subset_sha256': digest(manifest), 'target_provenance': provenance,
              'torch': torch.__version__, 'device': 'cpu', 'records': records, 'groups': groups,
              'all_predictions_reproduced': all(r['reproduces_saved_prediction'] for r in records)}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({'all_predictions_reproduced': result['all_predictions_reproduced'],
                      'graphs': len(records), 'train': groups['train']['nonempty'],
                      'valid': groups['valid']['nonempty']}, indent=2), flush=True)
    if not result['all_predictions_reproduced']:
        raise SystemExit('Recovered targets did not reproduce every saved prediction')


if __name__ == '__main__':
    main()
