"""Decode and score geometric graphs without using GT node counts at inference."""

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import torch


def decode_graph(output, threshold=0.5):
    h = output.h.detach().cpu().numpy()
    ids = np.flatnonzero(h > threshold)
    nodes = output.nodes.features.detach().cpu().numpy()[ids]
    adjacency = output.edges.adjacency.detach().cpu().numpy()[np.ix_(ids, ids)]
    adjacency = (adjacency + adjacency.T) / 2
    edges = np.argwhere(np.triu(adjacency > threshold, k=1))
    return {'nodes': nodes, 'edges': edges, 'node_confidence': h[ids],
            'edge_confidence': adjacency[edges[:, 0], edges[:, 1]], 'slots': ids}


def topology(graph):
    G = nx.Graph()
    G.add_nodes_from(range(len(graph['nodes'])))
    G.add_edges_from(graph['edges'].tolist())
    components = nx.number_connected_components(G)
    return components, G.number_of_edges() - G.number_of_nodes() + components


def score_graph(prediction, target, metric=None, tolerance=2.0):
    metric = np.eye(2) * 128 if metric is None else np.asarray(metric)
    pred_xy = np.asarray(prediction['nodes']) @ metric.T
    true_xy = np.asarray(target['nodes']) @ metric.T
    n_pred, n_true = len(pred_xy), len(true_xy)
    mapping = {}
    squared_error, assigned = 0., 0
    if n_pred and n_true:
        distances = np.linalg.norm(pred_xy[:, None] - true_xy[None, :], axis=-1)
        rows, cols = linear_sum_assignment(distances)
        errors = distances[rows, cols]
        squared_error = float((errors ** 2).sum())
        assigned = len(rows)
        mapping = {int(i): int(j) for i, j, d in zip(rows, cols, errors) if d <= tolerance}
    true_edges = {tuple(sorted(edge)) for edge in target['edges'].tolist()}
    edge_tp = sum(tuple(sorted((mapping[a], mapping[b]))) in true_edges
                  for a, b in prediction['edges'].tolist() if a in mapping and b in mapping)
    node_tp = len(mapping)
    edge_fp = len(prediction['edges']) - edge_tp
    edge_fn = len(true_edges) - edge_tp
    pc, pb = topology(prediction)
    tc, tb = topology(target)
    return {'n_true': n_true, 'n_pred': n_pred, 'node_tp': node_tp,
            'node_fp': n_pred - node_tp, 'node_fn': n_true - node_tp,
            'edge_tp': edge_tp, 'edge_fp': edge_fp, 'edge_fn': edge_fn,
            'squared_error': squared_error, 'assigned': assigned,
            'count_correct': n_pred == n_true,
            'exact': n_pred == n_true == node_tp and edge_fp == edge_fn == 0,
            'components_correct': pc == tc, 'cycles_correct': pb == tb,
            'coincident_target_nodes': len(np.unique(true_xy, axis=0)) < n_true}


def summarize(records):
    if not records:
        return {'graphs': 0}
    result = {'graphs': len(records)}
    for kind in ['node', 'edge']:
        tp, fp, fn = [sum(row[f'{kind}_{key}'] for row in records) for key in ['tp', 'fp', 'fn']]
        result[f'{kind}_precision'] = tp / max(tp + fp, 1)
        result[f'{kind}_recall'] = tp / max(tp + fn, 1)
        result[f'{kind}_f1'] = 2 * tp / max(2 * tp + fp + fn, 1)
    for key in ['count_correct', 'exact', 'components_correct', 'cycles_correct']:
        result[key] = float(np.mean([row[key] for row in records]))
    assigned = sum(row['assigned'] for row in records)
    result['coordinate_rmse_px'] = (sum(row['squared_error'] for row in records) / assigned) ** .5 if assigned else None
    result['assignment_coverage'] = assigned / max(sum(row['n_true'] for row in records), 1)
    result['false_positive_nodes'] = sum(row['node_fp'] for row in records)
    result['false_positive_edges'] = sum(row['edge_fp'] for row in records)
    result['coincident_target_graphs'] = sum(row['coincident_target_nodes'] for row in records)
    return result


def summarize_groups(records, samples):
    def group(indices):
        return summarize([records[i] for i in indices])
    result = {'all': summarize(records),
              'nonempty': group([i for i, row in enumerate(records) if row['n_true']]),
              'empty': group([i for i, row in enumerate(records) if not row['n_true']])}
    result['by_source'] = {source: group([i for i, row in enumerate(samples) if row['subject_id'] == source])
                           for source in sorted({row['subject_id'] for row in samples})}
    result['by_size'] = {name: group([i for i, row in enumerate(records) if low <= row['n_true'] <= high])
                         for name, low, high in [('0', 0, 0), ('1-8', 1, 8), ('9-16', 9, 16), ('17-32', 17, 32), ('33-64', 33, 64)]}
    return result


def sample_paths(paths, metric, step=0.5):
    points = []
    for path in paths:
        path = np.asarray(path) @ np.asarray(metric).T
        for a, b in zip(path[:-1], path[1:]):
            count = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
            points.append(np.linspace(a, b, count))
    return np.concatenate(points) if points else np.empty((0, 2))


def chord_paths(graph):
    return [graph['nodes'][edge] for edge in graph['edges']]


def path_distance(paths1, paths2, metric, tolerance=1.0):
    a, b = sample_paths(paths1, metric), sample_paths(paths2, metric)
    if not len(a) or not len(b):
        return {'f1': None if not len(a) and not len(b) else 0., 'mean_distance_px': None}
    da = cKDTree(b).query(a)[0]
    db = cKDTree(a).query(b)[0]
    precision, recall = np.mean(da <= tolerance), np.mean(db <= tolerance)
    return {'f1': float(2 * precision * recall / max(precision + recall, 1e-12)),
            'mean_distance_px': float((da.mean() + db.mean()) / 2)}


class GeometricGraphMetric:
    def __call__(self, outputs, targets, permutation_matrices=None):
        records = [score_graph(decode_graph(outputs[i]), decode_graph(targets[i])) for i in range(len(targets))]
        logs = {}
        for scope, rows in [('nonempty', [row for row in records if row['n_true']]),
                            ('empty', [row for row in records if not row['n_true']])]:
            logs.update({f'{scope}/{key}': value for key, value in summarize(rows).items()
                         if value is not None and key != 'graphs'})
        # This diagnostic uses GRALE's learned hard assignment, before external scoring.
        valid = targets.h
        if valid.any():
            error = ((outputs.nodes.features - targets.nodes.features) * 128).square().sum(-1)
            logs['matcher_coordinate_rmse_px'] = float((error[valid].mean()).sqrt())
        errors = [row['node_fp'] + row['node_fn'] + row['edge_fp'] + row['edge_fn'] for row in records]
        logs['edit_graph'] = float(np.mean(errors))
        return outputs.h.new_tensor(errors), logs
