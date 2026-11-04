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
    parser.add_argument('--analysis', type=Path)
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
    if args.analysis:
        analysis = json.loads(args.analysis.read_text())
        methods = ['learned', 'hungarian', 'fgw']
        x = list(range(3))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        values = [analysis['arms'][m]['connectivity_without_2px_node_gate']['train']['edge_f1'] for m in methods]
        bars = axes[0].bar(x, values, color=[colors[m] for m in methods], width=.6)
        axes[0].bar_label(bars, labels=[f'{v:.3f}' for v in values], padding=4)
        axes[0].set(title='Connectivity diagnostic', ylabel='Edge F1 without the 2-pixel node gate', ylim=(0, 1.13))
        own = [analysis['arms'][m]['same_count_input_swap']['own_centered_shape']['coordinate_rmse_px'] for m in methods]
        wrong = [analysis['arms'][m]['same_count_input_swap']['swapped_centered_shape']['coordinate_rmse_px'] for m in methods]
        own_bars = axes[1].bar([v-.17 for v in x], own, width=.32, color=[colors[m] for m in methods], label='Own input')
        wrong_bars = axes[1].bar([v+.17 for v in x], wrong, width=.32, color=[colors[m] for m in methods], alpha=.35, label='Swapped input')
        axes[1].bar_label(own_bars, labels=[f'{v:.1f}' for v in own], padding=3, fontsize=9)
        axes[1].bar_label(wrong_bars, labels=[f'{v:.1f}' for v in wrong], padding=3, fontsize=9)
        axes[1].set(title='Input-dependent shape: four same-count graphs', ylabel='Centered coordinate RMSE (pixels)', ylim=(0, 70))
        axes[1].legend(frameon=False, fontsize=9)
        for ax in axes:
            ax.set_xticks(x, ['Learned', 'Hungarian', 'FGW'])
            ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
        fig.suptitle('Saved-output mechanism diagnostics', fontsize=15)
        fig.text(.5, .025, 'Training-only, post-hoc diagnostics. They do not replace strict spatial reconstruction or establish generalization.',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .07, 1, .93))
        path = args.output.with_name(args.output.stem+'-diagnostics.png')
        fig.savefig(path, dpi=170)
        fig.savefig(path.with_suffix('.svg'))
        plt.close(fig)


if __name__ == '__main__':
    main()
