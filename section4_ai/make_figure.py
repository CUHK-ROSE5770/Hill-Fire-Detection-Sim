"""Draw Figure 3 from archived benchmark data and test box coordinates."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle


PALETTE = {'v8n_base': '#164e75', 'v8s_base': '#a44c23', 'v8n_aug': '#17796b'}
LABELS = {'v8n_base': 'v8n baseline', 'v8s_base': 'v8s baseline', 'v8n_aug': 'v8n augmented'}
MARKERS = {'v8n_base': 'o', 'v8s_base': 's', 'v8n_aug': '^'}
INK = '#182e3a'
SLATE = '#455e6d'


def pipeline(axes):
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.axis('off')
    labels = ('RGB\nYOLOv8', '3 matching\nframes', 'Aligned LWIR\nthermal rise', 'Ray–DEM\nlocalization', 'Mesh alert and\nverify / track')
    for index, label in enumerate(labels):
        x = .01 + index * .20
        axes.add_patch(FancyBboxPatch((x, .28), .176, .38, boxstyle='round,pad=.004,rounding_size=.018',
                                      facecolor='#e9f5f1' if index < 3 else '#edf3f7', edgecolor='#a5bbc5'))
        axes.text(x + .088, .47, label, ha='center', va='center', size=13, color=INK)
        if index < len(labels) - 1:
            axes.annotate('', (x + .197, .47), (x + .179, .47), arrowprops={'arrowstyle': '->', 'color': SLATE})
    axes.text(.01, .08, 'Without thermal corroboration, a persistent RGB cue stays a candidate for verification.',
              color=SLATE, size=12)


def title(axes, text):
    axes.set_title(text, loc='left', color=INK, fontweight='bold', fontsize=15, pad=9)


def qualitative(axes, cases):
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.axis('off')
    simple = sorted(cases, key=lambda case: len(case['truth']) + len(case['predictions']))
    hit = next((case for case in simple if any(pred[-2] == 0 and pred[-1] >= 0
                                               for pred in case['predictions'])), None)
    miss = next((case for case in simple if any(cls == 0 and small and i not in {
        pred[-1] for pred in case['predictions']} for i, (cls, *_, small) in enumerate(case['truth']))), None)
    negative = next((case for case in simple if not case['truth'] and case['predictions']), None)
    examples = [
        (hit, 'Smoke detected') if hit else (next(case for case in simple if case['truth']), 'Labeled positive frame'),
        (miss, 'Small smoke missed') if miss else (next(case for case in simple if case['truth']), 'Labeled positive frame'),
        (negative, 'Negative frame flagged') if negative else (next(case for case in simple if not case['truth']), 'Negative frame clear'),
    ]
    for index, (case, label) in enumerate(examples):
        origin_x = .015 + .332 * index
        width, height = .305, .55
        origin_y = .27
        axes.add_patch(Rectangle((origin_x, origin_y), width, height, facecolor='#edf4f5', edgecolor='#9eb5c1'))

        def box(coords, color, linestyle):
            x1, y1, x2, y2 = coords
            x = origin_x + width * x1 / case['width']
            y = origin_y + height * (1 - y2 / case['height'])
            dw = width * (x2 - x1) / case['width']
            dh = height * (y2 - y1) / case['height']
            axes.add_patch(Rectangle((x, y), dw, dh, fill=False, edgecolor=color, linewidth=2., linestyle=linestyle))

        for _, *coords, _ in case['truth']:
            box(coords, '#176e60', '-')
        for x1, y1, x2, y2, _, _, _ in case['predictions']:
            box((x1, y1, x2, y2), '#a44c23', '--')
        axes.text(origin_x + width / 2, .94, label, ha='center', va='center', fontsize=12, color=INK)
        axes.text(origin_x + width / 2, .18, case['filename'], ha='center', va='center', fontsize=9, color=SLATE)
    axes.legend(handles=[Line2D([], [], color='#176e60', lw=2, label='Reference box'),
                         Line2D([], [], color='#a44c23', lw=2, ls='--', label='Predicted box')],
                loc='lower center', ncol=2, frameon=False, fontsize=11)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='v8n_aug')
    args = parser.parse_args()
    results = json.loads((args.results / 'benchmark.json').read_text(encoding='utf-8'))['experiments']
    cases = json.loads((args.results / 'qualitative_boxes.json').read_text(encoding='utf-8'))
    figure = plt.figure(figsize=(13.3, 7.45), layout='constrained', facecolor='white')
    grid = figure.add_gridspec(3, 2, height_ratios=[.70, 1.50, 1.35], width_ratios=[1.60, 1.])
    design = figure.add_subplot(grid[0, :])
    title(design, '(a) Onboard detection and alert confirmation')
    pipeline(design)
    plot = figure.add_subplot(grid[1, 0])
    title(plot, '(b) Validation learning curves')
    for name in results:
        with (args.results / f'{name}_training.csv').open(newline='', encoding='utf-8') as stream:
            records = [{key.strip(): float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
        plot.plot([row['epoch'] for row in records], [row['metrics/mAP50(B)'] * 100 for row in records],
                  label=LABELS[name], color=PALETTE[name], marker=MARKERS[name], linewidth=2.3)
    plot.set(xlabel='Epoch', ylabel='mAP@0.5 (%)')
    plot.set_xticks(sorted({row['epoch'] for row in records}))
    plot.tick_params(labelsize=12)
    plot.xaxis.label.set_size(13)
    plot.yaxis.label.set_size(13)
    plot.grid(axis='y', alpha=.25)
    plot.spines[['right', 'top']].set_visible(False)
    plot.legend(loc='lower right', frameon=False, fontsize=12)

    matrix_axes = figure.add_subplot(grid[1, 1])
    title(matrix_axes, '(c) Held-out confusion counts')
    matrix = np.array(results[args.model]['test_confusion_true_by_predicted'])
    matrix_axes.imshow(np.log1p(matrix), cmap='Blues', vmin=0)
    tick_labels = ['Smoke', 'Fire', 'Background']
    matrix_axes.set_xticks(range(3), tick_labels, fontsize=11)
    matrix_axes.set_yticks(range(3), tick_labels, fontsize=11)
    matrix_axes.set(xlabel='Predicted', ylabel='Reference')
    for row, col in np.ndindex(matrix.shape):
        color = 'white' if np.log1p(matrix[row, col]) > np.log1p(matrix.max()) * .57 else INK
        count = '–' if (row, col) == (2, 2) else str(matrix[row, col])
        matrix_axes.text(col, row, count, color=color, va='center', ha='center', size=14)

    bottom = figure.add_subplot(grid[2, :])
    title(bottom, '(d) Selected held-out detections')
    qualitative(bottom, cases)
    figure.savefig(args.output, dpi=220, facecolor='white')


if __name__ == '__main__':
    main()
