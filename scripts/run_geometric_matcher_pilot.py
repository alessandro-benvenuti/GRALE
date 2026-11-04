"""Paired learned Sinkhorn / geometric Hungarian / partial-FGW reconstruction fits."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from GRALE.data.dataset import custom_collate_fn
from GRALE.model.geometric_assignment import AssignmentForward
from scripts.inspect_geometric_run import digest
from scripts.run_spatial_anchor_pilot import initialize, inspection_datasets, evaluate, plot_comparison

METHODS = ['learned', 'hungarian', 'fgw']


def snapshot(model, datasets, method, step, arm_dir, predict):
    predict.capture = []
    result = evaluate(model, datasets, method, step, arm_dir, forward_fn=predict)
    assert len(predict.capture) == len(result['per_graph'])
    folder = arm_dir / 'snapshots' / f'step-{step:06d}'
    for row, (stats, details) in zip(result['per_graph'], predict.capture):
        row['assignment_diagnostics'] = stats
        if method == 'fgw':
            detail = details[0]
            row['fgw_solver'] = detail.get('solver_diagnostic', {'termination_evidence': 'empty_target'})
            np.savez_compressed(folder / f'fgw-soft-{row["split"]}-{row["sample_id"]}.npz',
                                soft_transport=detail['soft_transport'], candidate_ids=detail['candidate_ids'])
    predict.capture = None
    (folder / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path)
    parser.add_argument('--manifest', type=Path, help='Local generated fixture only')
    parser.add_argument('--config', type=Path, default=ROOT / 'GRALE/configs/octa_straight_l1_200k.yaml')
    parser.add_argument('--inspection-targets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    args = parser.parse_args()
    if bool(args.reference_run) == bool(args.manifest) or args.steps < 1 or len(set(args.methods)) != len(args.methods):
        parser.error('Select one reference/fixture, positive steps and distinct methods')
    torch.set_num_threads(1)
    if 'fgw' in args.methods:
        if importlib.metadata.version('POT') != '0.9.7':
            raise ValueError('This frozen experiment requires POT 0.9.7')
        from ot.gromov import partial_fused_gromov_wasserstein
    if args.reference_run:
        reference = json.loads((args.reference_run / 'result.json').read_text())
        manifest = Path(reference['subset'])
        if digest(manifest) != reference['subset_sha256']:
            raise ValueError('Reference cohort changed')
        config = yaml.safe_load((args.reference_run / 'config.yaml').read_text())
    else:
        manifest = args.manifest
        config = yaml.safe_load(args.config.read_text())
    frozen = json.loads(manifest.read_text())
    datasets, provenance = inspection_datasets(manifest, frozen, config,
                                               args.inspection_targets, args.reference_run)
    if len(datasets['train']) != 10 or len(datasets['valid']) != 4:
        raise ValueError('Use the frozen 10/4 cohort')
    if {r['subject_id'] for r in datasets['train'].samples} & {r['subject_id'] for r in datasets['valid'].samples}:
        raise ValueError('Train and validation sources overlap')
    if any(r['nodes'] > 20 for d in datasets.values() for r in d.samples):
        raise ValueError('GT exceeds the max-20 cutoff')
    config.update(n_grad_steps=args.steps, n_warmup_steps=min(50, args.steps // 10),
                  matcher_model='sinkhorn', matcher_cost='l1', batchsize_effective=4,
                  assignment_class_cost=2., assignment_node_cost=5., assignment_fgw_alpha=.4,
                  assignment_candidate_count=32)
    if config['node_pos_dim'] != 0 or config['node_features_dim'] != 2 or config['n_nodes_max'] != 64:
        raise ValueError('Expected coordinate-only XY and original 64-slot model')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = {'purpose': 'Controlled correspondence experiment, one seed and tiny cohort',
             'reference_run': str(args.reference_run), 'subset': str(manifest),
             'subset_sha256': digest(manifest), 'target_provenance': provenance,
             'config': config, 'torch': torch.__version__, 'device': args.device,
             'pot': importlib.metadata.version('POT') if 'fgw' in args.methods else None,
             'pot_package_provenance_sha256': digest(Path(os.environ['GRALE_SOLVER_DEPS']) / 'PACKAGE_PROVENANCE.json')
                 if os.environ.get('GRALE_SOLVER_DEPS') else None,
             'source_commit': (ROOT / 'SOURCE_COMMIT').read_text().strip() if (ROOT / 'SOURCE_COMMIT').exists() else None,
             'source_archive_sha256': (ROOT / 'SOURCE_ARCHIVE_SHA256').read_text().strip() if (ROOT / 'SOURCE_ARCHIVE_SHA256').exists() else None,
             'code_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                             [*sorted((ROOT / 'GRALE').rglob('*.py')), Path(__file__).resolve(),
                              ROOT / 'scripts/run_spatial_anchor_pilot.py']},
             'arms': {}}
    milestones = {s for s in [0, 50, 200, 500, 1000, 2000, 3000, args.steps] if s <= args.steps}
    parameter_hash = order_hash = None
    for method in args.methods:
        arm_dir = output / method
        arm_dir.mkdir()
        method_config = dict(config, loss_assignment=method)
        model, arm_config, _, full_sha = initialize(method_config, 'default', args.device)
        if parameter_hash is not None and parameter_hash != full_sha:
            raise AssertionError('Initial model parameters differ across arms')
        parameter_hash = full_sha
        predict = AssignmentForward(method)
        optimizers, schedulers = model.configure_optimizers()
        optimizer, scheduler = optimizers[0], schedulers[0]['scheduler']
        snapshots = [snapshot(model, datasets, method, 0, arm_dir, predict)]
        loader = DataLoader(list(range(len(datasets['train']))), batch_size=4, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(config['seed']),
                            collate_fn=lambda ids: (ids, custom_collate_fn([datasets['train'][i] for i in ids])))
        iterator = iter(loader)
        order = hashlib.sha256()
        started = time.time()
        match_seconds = changed = real_batches = warnings = 0.
        with (arm_dir / 'losses.csv').open('w', newline='') as stream:
            writer = None
            for step in range(1, args.steps + 1):
                try:
                    ids, batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    ids, batch = next(iterator)
                order.update((json.dumps(ids) + '\n').encode())
                optimizer.zero_grad(set_to_none=True)
                _, _, loss, terms = predict(model, batch.to(args.device))
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite loss: {method}, {step}')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'], error_if_nonfinite=True)
                row = {'step': step, 'batch_size': len(ids), 'learning_rate': optimizer.param_groups[0]['lr'],
                       'gradient_norm': float(norm), **terms, **predict.stats}
                optimizer.step()
                scheduler.step()
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                match_seconds += predict.stats['matching_seconds']
                if method == 'fgw':
                    for details in model.geometric_assignment.last_details:
                        changed += details['changed_targets_vs_unary']
                        real_batches += details['real_targets']
                warnings += predict.stats['fgw_warning_count']
                if step in milestones:
                    stream.flush()
                    snapshots.append(snapshot(model, datasets, method, step, arm_dir, predict))
                    torch.save({'state_dict': model.state_dict(), 'hyper_parameters': arm_config,
                                'pytorch-lightning_version': pl.__version__, 'hparams_name': 'config',
                                'global_step': step, 'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict()}, arm_dir / 'last.ckpt')
                    (arm_dir / 'snapshots.json').write_text(json.dumps(snapshots, indent=2, allow_nan=False) + '\n')
        if order_hash is not None and order_hash != order.hexdigest():
            raise AssertionError('Training order differs across arms')
        order_hash = order.hexdigest()
        state['arms'][method] = {'initial_parameter_sha256': full_sha, 'minibatch_order_sha256': order_hash,
                                  'steps': args.steps, 'seconds': time.time() - started,
                                  'effective_weight_decay': optimizer.param_groups[0]['weight_decay'],
                                  'matching_host_seconds': match_seconds,
                                  'changed_targets_vs_unary_fraction': changed / max(real_batches, 1),
                                  'solver_warning_count': warnings, 'snapshots': snapshots}
        (output / 'result.json').write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
        plot_comparison(state['arms'], output)
        del optimizer, scheduler, model
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    print('GEOMETRIC_MATCHER_PILOT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
