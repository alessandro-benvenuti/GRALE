"""Plot aggregate pilot results; no individual graph data are required."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('output', type=Path, help='Output PNG path; an SVG is saved alongside it')
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    labels = {'default': 'Original', 'center': 'Centered control',
              'spread': 'Spatial anchors', 'spread_warmup': 'Anchors + warmup'}
    colors = {'default': '#3974ac', 'center': '#777777',
              'spread': '#dd8926', 'spread_warmup': '#128d80'}
    for arm, state in summary['arms'].items():
        for ax, key in zip(axes.flat, ['coordinate_rmse_px', 'median_spread_ratio',
                                      'effective_output_slots_per_target', 'node_f1']):
            source = 'train' if key in ['coordinate_rmse_px', 'node_f1'] else 'train_diagnostics'
            points = [(r['step'], r[source][key]) for r in state['snapshots']
                      if r[source][key] is not None and
                      (key != 'coordinate_rmse_px' or
                       (r['train']['count_correct'] == 1 and r['train']['assignment_coverage'] == 1))]
            if points:
                x, y = zip(*points)
                ax.plot(x, y, '.-', color=colors[arm], label=labels[arm], lw=1.8, ms=6)
    settings = [('Localization error', 'RMSE (pixels)'),
                ('Selected nodes collapse after release', 'Median selected-node / GT spread'),
                ('Matching remains diffuse', 'Effective output slots per GT node'),
                ('Localization gains do not persist', 'Node F1 (2-pixel tolerance)')]
    for ax, (title, ylabel) in zip(axes.flat, settings):
        ax.set(title=title, xlabel='Optimizer update', ylabel=ylabel, xlim=(0, 5100))
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 1].set(yscale='log', ylim=(.003, 1.5))
    axes[0, 1].axhline(1, color='black', ls=':', lw=1)
    axes[0, 1].axvspan(0, 500, color=colors['spread_warmup'], alpha=.09)
    axes[0, 1].axvspan(500, 1000, color=colors['spread_warmup'], alpha=.04)
    axes[1, 0].set(yscale='log', ylim=(.8, 75))
    axes[1, 0].axhline(1, color='black', ls=':', lw=1)
    axes[1, 1].set(ylim=(-.001, .022))
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.suptitle('GRALE spatial-initialization pilot', fontsize=16, y=.995)
    fig.text(.5, .955, 'Nonempty training metrics (8 graphs) | frozen 10/4 cohort | seed 1234',
             ha='center', fontsize=10)
    fig.legend(handles, names, loc='upper center', bbox_to_anchor=(.5, .936), ncol=4, frameon=False)
    fig.text(.5, .014, 'Warmup shading: hold 0–500, ramp 500–1000. RMSE shown only with correct counts and full matching coverage.',
             ha='center', fontsize=8.5)
    fig.tight_layout(rect=(0, .04, 1, .895))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    fig.savefig(args.output.with_suffix('.svg'))
    plt.close(fig)


if __name__ == '__main__':
    main()
