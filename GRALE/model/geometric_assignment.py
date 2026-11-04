"""Hard XY/graph correspondence for the controlled reconstruction pilot.

Matching is detached; the existing losses still differentiate native outputs.
The original learned matcher remains in the model for paired initialization,
but is bypassed in the two geometric-assignment arms.
"""
import time

import torch

from .gnbm_matcher_reference import HungarianMatcher, FusedGromovWassersteinMatcher


class GeometricAssignment:
    def __init__(self, method, *, class_cost=2., node_cost=5., alpha=.4,
                 candidate_count=32, max_iter=10000, tolerance=1e-7):
        if method not in ('hungarian', 'fgw'):
            raise ValueError('Geometric assignment must be hungarian or fgw')
        self.method = method
        self.unary = HungarianMatcher(class_cost, node_cost, dimensions=2)
        self.fgw = FusedGromovWassersteinMatcher(
            class_cost, node_cost, alpha, dimensions=2, candidate_count=candidate_count,
            max_iter=max_iter, tolerance=tolerance) if method == 'fgw' else None
        self.last_details = None

    @staticmethod
    def complete_permutation(assignments, real_columns, size, like):
        """Rows are decoder slots; columns are original, padded GT node IDs."""
        matrix = like.new_zeros((len(assignments), size, size))
        for batch, ((sources, targets), columns) in enumerate(zip(assignments, real_columns)):
            sources = sources.to(like.device, dtype=torch.long)
            targets = targets.to(like.device, dtype=torch.long)
            columns = columns.to(like.device, dtype=torch.long)
            n = len(columns)
            if (len(sources) != n or len(targets) != n or sources.unique().numel() != n
                    or not torch.equal(targets.sort().values, torch.arange(n, device=like.device))
                    or (n and (sources.min() < 0 or sources.max() >= size))):
                raise ValueError('Hard assignment must match each GT node to a distinct valid slot')
            matrix[batch, sources, columns[targets]] = 1
            free_sources = torch.ones(size, dtype=torch.bool, device=like.device)
            free_columns = free_sources.clone()
            free_sources[sources] = False
            free_columns[columns] = False
            matrix[batch, free_sources.nonzero().flatten(), free_columns.nonzero().flatten()] = 1
        if not (torch.all(matrix.sum(1) == 1) and torch.all(matrix.sum(2) == 1)):
            raise ValueError('Completed assignment is not a permutation')
        return matrix

    @torch.no_grad()
    def __call__(self, output, target):
        started = time.perf_counter()
        real_columns = [mask.nonzero().flatten() for mask in target.h]
        nodes, edges = [], []
        for b, columns in enumerate(real_columns):
            nodes.append(target.nodes.features[b, columns])
            adjacency = target.edges.adjacency[b][columns][:, columns]
            edges.append(torch.triu(adjacency, diagonal=1).nonzero())
        # softmax([0, h])_object == sigmoid(h), matching the binary GRALE head.
        outputs = {'pred_nodes': output.nodes.features,
                   'pred_logits': torch.stack((torch.zeros_like(output.h), output.h), dim=-1)}
        targets = {'nodes': nodes, 'edges': edges}
        unary = self.unary(outputs, targets)
        details = []
        if self.method == 'hungarian':
            assignments = unary
            for columns in real_columns:
                details.append({'real_targets': len(columns), 'changed_targets_vs_unary': 0,
                                'solver_iterations': 0, 'solver_warnings': []})
        else:
            candidates = self.fgw.matching_candidates(outputs, targets)
            probabilities = output.edges.adjacency.sigmoid()
            probabilities = (probabilities + probabilities.transpose(1, 2)) / 2
            structures = [probabilities[b][ids][:, ids] for b, ids in enumerate(candidates)]
            assignments, transports, diagnostics = self.fgw(
                outputs, targets, candidate_indices=candidates, predicted_structure=structures,
                return_transport=True, return_diagnostics=True)
            for columns, ids, assigned, initial, coupling, diagnostic in zip(
                    real_columns, candidates, assignments, unary, transports, diagnostics):
                warnings = diagnostic.get('warnings', [])
                if warnings or diagnostic.get('emd_warning'):
                    raise RuntimeError(f'FGW solver warning; refusing silent fallback: {diagnostic}')
                n = len(columns)
                a = assigned[0][assigned[1].argsort()]
                u = initial[0][initial[1].argsort()]
                details.append({'real_targets': n, 'changed_targets_vs_unary': int((a != u).sum()),
                                'candidate_ids': ids.cpu().numpy(), 'soft_transport': coupling,
                                'solver_iterations': diagnostic.get('iterations', 0),
                                'solver_warnings': warnings, 'solver_diagnostic': diagnostic})
        matrix = self.complete_permutation(assignments, real_columns, output.h.shape[1], output.h)
        self.last_details = details
        n = sum(row['real_targets'] for row in details)
        stats = {'assignment_hard': 1., 'matching_seconds': time.perf_counter() - started,
                 'changed_target_fraction_vs_unary': sum(d['changed_targets_vs_unary'] for d in details) / max(n, 1),
                 'fgw_max_iterations': max((d['solver_iterations'] for d in details), default=0),
                 'fgw_warning_count': sum(len(d['solver_warnings']) for d in details)}
        return matrix, stats


class AssignmentForward:
    """Native predictions plus the chosen loss correspondence; never GT-count decoding."""
    def __init__(self, method):
        if method not in ('learned', 'hungarian', 'fgw'):
            raise ValueError('Unknown assignment strategy')
        self.method = method
        self.stats = {}
        self.capture = None

    def __call__(self, model, batch):
        if model.hparams.get('loss_assignment', 'learned') != self.method:
            raise ValueError('Model and experiment assignment strategies differ')
        inputs = model.input_builder(batch.clone())
        encoded, latent = model.encoder(inputs)
        embeddings, output = model.decoder(latent)
        target = model.target_builder(batch.clone())
        started = time.perf_counter()
        if self.method == 'learned':
            T, _ = model.matcher(encoded, ~inputs.h, embeddings)
            self.stats = {'assignment_hard': 0., 'matching_seconds': time.perf_counter() - started,
                          'changed_target_fraction_vs_unary': 0., 'fgw_max_iterations': 0,
                          'fgw_warning_count': 0}
            details = None
        else:
            if model.geometric_assignment is None or model.geometric_assignment.method != self.method:
                raise ValueError('Checkpoint/config assignment strategy does not match the runner')
            T, self.stats = model.match_graphs(encoded, inputs, embeddings, output, targets=target)
            details = model.geometric_assignment.last_details
        loss, terms = model.training_objective(output, target, T)
        if self.capture is not None:
            self.capture.append((dict(self.stats), details))
        return output, T, loss.mean(), terms
