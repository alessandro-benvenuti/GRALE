"""Post-hoc connectivity and input-dependence checks using saved arrays only.

No model loading, inference, optimization or threshold selection. Strict metrics
remain primary; unrestricted spatial correspondence is a labeled diagnostic.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from GRALE.loss.geometry import score_graph, summarize
from scripts.inspect_geometric_run import digest


def centered(graph):
    nodes = graph['nodes']
    return dict(graph, nodes=nodes-nodes.mean(0) if len(nodes) else nodes.copy())


def same_count_swaps(samples):
    counts = defaultdict(list)
    for sample in samples:
        if sample['split']=='train' and sample['nodes']:
            counts[sample['nodes']].append(sample['sample_id'])
    swaps = {}
    for group in counts.values():
        if len(group)>1:
            for i, sample in enumerate(group):
                swaps[sample] = group[(i+1)%len(group)]
    return swaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-step', type=int, default=5000)
    args = parser.parse_args()
    state = json.loads((args.run / 'result.json').read_text())
    assert set(state['arms']) == {'learned','hungarian','fgw'}
    manifest = Path(state['subset'])
    assert digest(manifest) == state['subset_sha256']
    frozen = json.loads(manifest.read_text())
    inspection = Path(state['target_provenance']['report']).parent
    assert digest(inspection/'report.json') == state['target_provenance']['report_sha256']
    for path, sha in state['target_provenance']['target_artifacts_sha256'].items():
        assert digest(path) == sha
    swaps = same_count_swaps(frozen['samples'])
    result = {'purpose':'Post-hoc mechanism diagnostics; strict 2px metrics remain primary',
              'subset_sha256':digest(manifest), 'run':str(args.run),
              'input_swap_rule':'Cyclic swap within nonempty training groups with equal GT node counts; frozen manifest order',
              'swap_graph_count':len(swaps), 'arms':{}}
    for method, arm in state['arms'].items():
        endpoint = arm['snapshots'][-1]
        assert endpoint['step']==args.expected_step
        snapshots = args.run/method/'snapshots'/f'step-{args.expected_step:06d}'
        predictions, targets, metrics = {}, {}, {}
        for sample in frozen['samples']:
            split = 'valid' if sample['split']=='validation' else sample['split']
            key = (split,sample['sample_id'])
            with np.load(snapshots/f'{split}-{sample["sample_id"]}.npz',allow_pickle=False) as saved:
                predictions[key] = {k:saved[k] for k in ['nodes','edges']}
            with np.load(inspection/f'{split}-{sample["sample_id"]}.npz',allow_pickle=False) as saved:
                targets[key] = {'nodes':saved['target_nodes'],'edges':saved['target_edges']}
                metrics[key] = saved['metric']
        summaries = {}
        for split in ['train','valid']:
            records=[]
            for key,target in targets.items():
                if key[0]!=split or not len(target['nodes']):
                    continue
                strict = score_graph(predictions[key],target,metrics[key],tolerance=2.)
                reference = next(r for r in endpoint['per_graph'] if (r['split'],r['sample_id'])==key)
                for field in ['node_tp','node_fp','node_fn','edge_tp','edge_fp','edge_fn','n_true','n_pred']:
                    assert strict[field]==reference[field],(method,key,field)
                records.append(score_graph(predictions[key],target,metrics[key],tolerance=float('inf')))
            score = summarize(records)
            summaries[split] = {k:score[k] for k in ['graphs','edge_precision','edge_recall','edge_f1','assignment_coverage']}
        own, wrong, own_shape, wrong_shape = [],[],[],[]
        for sample, other in swaps.items():
            key=('train',sample)
            source=('train',other)
            gt, metric=targets[key],metrics[key]
            own.append(score_graph(predictions[key],gt,metric))
            wrong.append(score_graph(predictions[source],gt,metric))
            own_shape.append(score_graph(centered(predictions[key]),centered(gt),metric))
            wrong_shape.append(score_graph(centered(predictions[source]),centered(gt),metric))
        swapped = {name:{k:v for k,v in summarize(rows).items() if k in
                        ['graphs','coordinate_rmse_px','assignment_coverage','count_correct']}
                   for name,rows in [('own_input',own),('swapped_same_count_input',wrong),
                                     ('own_centered_shape',own_shape),('swapped_centered_shape',wrong_shape)]}
        fg = [r.get('fgw_solver',{}) for r in endpoint['per_graph'] if r['n_true']]
        result['arms'][method] = {'connectivity_without_2px_node_gate':summaries,
                                  'same_count_input_swap':swapped,
                                  'fgw_changed_targets_vs_unary_fraction':arm['changed_targets_vs_unary_fraction'],
                                  'fgw_endpoint_solver_diagnostics':fg if method=='fgw' else []}
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({'swap_graph_count':len(swaps),'arms':result['arms']},indent=2,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
