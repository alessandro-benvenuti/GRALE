"""One-graph matching comparison. This measures memorization, not generalization.

From the GRALE root, in the existing PyTorch environment:
  python scripts/diagnose_synthetic_matching.py --device cuda
Use --steps 2 --device cpu --output /tmp/grale-matching-smoke for a cheap check.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import shortest_path
import torch
import yaml

from GRALE.data.dense_data import BatchedDenseData
from GRALE.data.dataset import custom_collate_fn
from GRALE.data.geometry import GeometricDataset
from GRALE.loss.geometry import decode_graph, score_graph
from GRALE.main import GRALE_model


def synthetic_graph(capacity, device):
    xy = torch.tensor([[.18, .22], [.48, .76], [.84, .32]], device=device)
    h = torch.arange(capacity, device=device)[None] < len(xy)
    features = torch.zeros(1, capacity, 2, device=device)
    features[0, :len(xy)] = xy
    adjacency = torch.zeros(1, capacity, capacity, device=device)
    for a, b in [(0, 1), (1, 2)]:
        adjacency[0, a, b] = adjacency[0, b, a] = 1
    distances = torch.full_like(adjacency, float(capacity))
    distances[0, :3, :3] = torch.as_tensor(shortest_path(
        adjacency[0, :3, :3].cpu().numpy(), directed=False, unweighted=True), device=device)
    return BatchedDenseData(h, {'features': features}, {
        'adjacency': adjacency, 'SP': distances,
        'labels': torch.nn.functional.one_hot(adjacency.long(), 2).float()})


def forward(model, batch, arm):
    encoded, latent = model.encoder(model.input_builder(batch.clone()))
    decoded, output = model.decoder(latent)
    # Fixed identity supervises the active input slots and all padding slots.
    # It supplies correspondence only to the loss, never to the decoder.
    transport = None if arm == 'fixed' else model.matcher(encoded, ~batch.h, decoded)[0]
    targets = model.target_builder(batch)
    loss, terms = model.training_objective(output, targets, transport)
    return output, transport, loss.mean(), terms


def snapshot(model, batch, arm, step, output_dir, terms):
    model.eval()
    with torch.no_grad():
        output, transport, _, _ = forward(model, batch, arm)
        prediction = decode_graph(model.format_logits(output)[0])
        target = decode_graph(batch[0])
        record = score_graph(prediction, target)
        record.update(step=step, arm=arm, **terms)
        record['coordinate_rmse_px'] = ((record['squared_error'] / record['assigned']) ** .5
                                         if record['assigned'] else None)
        def spread(x):
            return float(np.sqrt(((x - x.mean(0)) ** 2).sum(1).mean()) * 128) if len(x) else None
        record['predicted_spread_px'] = spread(prediction['nodes'])
        record['target_spread_px'] = spread(target['nodes'])
        if transport is not None:
            real = transport[0, :, :int(batch.h[0].sum())]
            conditional = real / real.sum(0, keepdim=True)
            record['mean_max_assignment_weight'] = float(conditional.max(0).values.mean())
            record['mean_assignment_entropy'] = float(-(
                conditional * conditional.clamp_min(1e-30).log()).sum(0).mean())
            record['marginal_max_error'] = float(torch.maximum(
                (transport.sum(1) - 1).abs().max(), (transport.sum(2) - 1).abs().max()))
            np.save(output_dir / f'transport-{step:06d}.npy', transport[0].cpu().numpy())
        np.savez_compressed(output_dir / f'prediction-{step:06d}.npz', **prediction)
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for ax, graph, title in zip(axes, [target, prediction], ['Target', arm]):
            for edge in graph['edges']:
                points = graph['nodes'][edge] * 128
                ax.plot(points[:, 0], points[:, 1], color='teal')
            ax.scatter(graph['nodes'][:, 0] * 128, graph['nodes'][:, 1] * 128,
                       s=24, facecolors='none', edgecolors='red')
            ax.set(xlim=(-10, 138), ylim=(138, -10), aspect='equal',
                   title=f'{title}: {len(graph["nodes"])} nodes, {len(graph["edges"])} edges')
        fig.suptitle(f'Step {step}; coordinate RMSE: {record["coordinate_rmse_px"]}')
        fig.tight_layout()
        fig.savefig(output_dir / f'prediction-{step:06d}.png', dpi=120)
        plt.close(fig)
    model.train()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--manifest', type=Path, help='Prepared manifest; omit for the synthetic graph')
    parser.add_argument('--sample-id', help='Exact training sample from --manifest')
    args = parser.parse_args()
    if bool(args.manifest) != bool(args.sample_id):
        parser.error('--manifest and --sample-id must be supplied together')
    if args.output is None:
        args.output = Path('logs/real-single-matching-20261029' if args.manifest
                           else 'logs/synthetic-matching-20261029')
    if args.output.exists():
        raise ValueError('Choose a new output directory to preserve existing evidence')
    if args.steps < 1:
        parser.error('--steps must be positive')
    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    config = yaml.safe_load((ROOT / 'GRALE/configs/octa_straight.yaml').read_text())
    config.update(n_grad_steps=args.steps, n_warmup_steps=min(50, args.steps // 10),
                  batchsize_effective=1)
    if args.manifest:
        dataset = GeometricDataset(args.manifest, 'train', config['n_nodes_max'])
        index = next((i for i, row in enumerate(dataset.samples)
                      if row['sample_id'] == args.sample_id), None)
        if index is None:
            raise ValueError('Requested sample is not in the training split')
        raw = dataset.load_target(index)
        if not np.allclose(raw['metric'], np.eye(2) * 128) or tuple(raw['shape_hw']) != (128, 128):
            raise ValueError('This diagnostic uses 128x128 pixel coordinates')
        batch = custom_collate_fn([dataset[index]]).to(args.device)
        source = {'kind': 'prepared dataset graph', 'sample': dataset.samples[index],
                  'manifest': str(args.manifest.resolve()),
                  'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest()}
    else:
        batch = synthetic_graph(config['n_nodes_max'], args.device)
        source = {'kind': 'synthetic bent path'}
    target = decode_graph(batch[0])
    if not len(target['nodes']):
        raise ValueError('This diagnostic requires a nonempty graph')
    np.savez_compressed(args.output / 'target.npz', **decode_graph(batch[0]))
    state = {'purpose': 'Same single graph, three matching methods; no generalization claim',
             'device': args.device, 'torch': torch.__version__, 'config': config,
             'node_order': 'Fixed for every update; no permutation augmentation',
             'graph': {'nodes': target['nodes'].tolist(), 'edges': target['edges'].tolist()},
             'source': source,
             'code_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
             'code_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [Path(__file__).resolve(), ROOT / 'GRALE/model/matcher.py']},
             'arms': {}}
    schedule_steps = {0, 50, 200, 500, 1000, args.steps}
    for arm in ['l1', 'squared_l2', 'fixed']:
        torch.manual_seed(config['seed'])
        if args.device == 'cuda':
            torch.cuda.manual_seed_all(config['seed'])
        arm_config = dict(config, matcher_cost='squared_l2' if arm == 'squared_l2' else 'l1')
        model = GRALE_model(arm_config).to(args.device)
        initial_sha = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes()
                                             for p in model.parameters())).hexdigest()
        optimizers, schedulers = model.configure_optimizers()
        optimizer = optimizers[0]
        scheduler = schedulers[0]['scheduler']
        arm_dir = args.output / arm
        arm_dir.mkdir()
        records = [snapshot(model, batch, arm, 0, arm_dir, {})]
        losses = []
        started = time.time()
        model.train()
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            output, transport, loss, terms = forward(model, batch, arm)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss: {arm}, step {step}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'],
                                                  error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            losses.append(dict(step=step, gradient_norm=float(norm), **terms))
            if step in schedule_steps:
                records.append(snapshot(model, batch, arm, step, arm_dir, terms))
                (arm_dir / 'snapshots.json').write_text(json.dumps(records, indent=2) + '\n')
                print(json.dumps(records[-1]), flush=True)
        torch.save({'state_dict': model.state_dict(), 'hyper_parameters': arm_config,
                    'global_step': args.steps}, arm_dir / 'final.ckpt')
        (arm_dir / 'losses.json').write_text(json.dumps(losses) + '\n')
        final = records[-1]
        state['arms'][arm] = {'initial_parameter_sha256': initial_sha,
            'effective_weight_decay': optimizer.param_groups[0]['weight_decay'],
            'seconds': time.time() - started, 'final': final,
            'passed': final['exact'] and final['coordinate_rmse_px'] <= 1}
        (args.output / 'result.json').write_text(json.dumps(state, indent=2) + '\n')
    assert len({v['initial_parameter_sha256'] for v in state['arms'].values()}) == 1
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for arm in state['arms']:
        losses = json.loads((args.output / arm / 'losses.json').read_text())
        records = json.loads((args.output / arm / 'snapshots.json').read_text())
        axes[0].plot([r['step'] for r in losses], [r['loss'] for r in losses], label=arm)
        valid = [r for r in records if r['coordinate_rmse_px'] is not None]
        axes[1].plot([r['step'] for r in valid], [r['coordinate_rmse_px'] for r in valid], 'o-', label=arm)
    axes[0].set(xlabel='Update', ylabel='Training loss', yscale='log')
    axes[1].set(xlabel='Update', ylabel='Assigned-node RMSE (pixels)')
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / 'learning-curves.png', dpi=140)
    plt.close(fig)
    print(json.dumps(state['arms'], indent=2), flush=True)


if __name__ == '__main__':
    main()
