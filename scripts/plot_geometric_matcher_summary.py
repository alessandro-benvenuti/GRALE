"""Plot aggregate learned/Hungarian/FGW training trajectories."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    labels = {'learned': 'Learned L1 Sinkhorn', 'hungarian': 'Geometric Hungarian', 'fgw': 'Partial FGW + hardening'}
    colors = {'learned': '#3974ac', 'hungarian': '#dd8926', 'fgw': '#128d80'}
    keys = ['node_f1', 'edge_f1', 'coordinate_rmse_px', 'count_correct']
    for method, state in summary['arms'].items():
        for ax, key in zip(axes.flat, keys):
            points = [(r['step'], r['train'][key]) for r in state['snapshots'] if r['train'][key] is not None]
            if points:
                x, y = zip(*points)
                ax.plot(x, y, '.-', color=colors[method], label=labels[method], lw=1.8, ms=6)
    settings = [('Node reconstruction', 'Node F1 (2-pixel tolerance)'),
                ('Edge reconstruction', 'Edge F1'),
                ('Localization error', 'RMSE on assigned pairs (pixels)'),
                ('Node-count prediction', 'Fraction of graphs with correct count')]
    for index, (ax, (title, ylabel)) in enumerate(zip(axes.flat, settings)):
        ax.set(title=title, xlabel='Optimizer update', ylabel=ylabel, xlim=(0, 5100))
        ax.set_ylim(bottom=0 if index == 2 else -.02)
        if index != 2:
            ax.set_ylim(top=1.02)
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.suptitle('GRALE correspondence comparison', fontsize=16, y=.995)
    fig.text(.5, .955, 'Nonempty training metrics (8 graphs) | same initialization and 10/4 cohort | seed 1234',
             ha='center', fontsize=10)
    fig.legend(handles, names, loc='upper center', bbox_to_anchor=(.5, .936), ncol=3, frameon=False)
    fig.text(.5, .014, 'Native thresholded predictions; no GT-count decoding. Interpret assigned-pair RMSE together with coverage and F1.',
             ha='center', fontsize=8.5)
    fig.tight_layout(rect=(0, .04, 1, .895))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    fig.savefig(args.output.with_suffix('.svg'))
    plt.close(fig)


if __name__ == '__main__':
    main()
