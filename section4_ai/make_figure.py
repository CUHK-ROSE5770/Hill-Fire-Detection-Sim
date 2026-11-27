"""Draw Figure 3 from archived benchmark data and test box coordinates."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image, ImageOps


PALETTE = {'v8n_base': '#164e75', 'v8s_base': '#a44c23', 'v8n_aug': '#17796b'}
LABELS = {'v8n_base': 'v8n baseline', 'v8s_base': 'v8s baseline', 'v8n_aug': 'v8n augmented'}
MARKERS = {'v8n_base': 'o', 'v8s_base': 's', 'v8n_aug': '^'}
INK = '#182e3a'
SLATE = '#455e6d'
CLASS_NAMES = ('smoke', 'fire')
REFERENCE_COLOR = '#69edb6'
PREDICTION_COLOR = '#ffb454'
EXAMPLES = (
    ('AoF07925.jpg', 'Hillside fire', 'fire'),
    ('AoF07976.jpg', 'Smoke plume', 'smoke'),
    ('AoF07986.jpg', 'Small smoke plume', 'smoke'),
    ('AoF06871.jpg', 'Background frame', None),
)


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


def qualitative(axes, cases, images):
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.axis('off')
    by_filename = {case['filename']: case for case in cases}
    for index, (filename, label, target_class) in enumerate(EXAMPLES):
        case = by_filename[filename]
        origin_x = .008 + .25 * index
        center = origin_x + .117
        width, height = case['width'], case['height']
        frame = axes.inset_axes([origin_x, .24, .234, .60])
        with Image.open(images / filename) as source:
            photo = ImageOps.exif_transpose(source).convert('RGB')
            if photo.size != (width, height):
                raise ValueError(f'Image size does not match saved detection coordinates for {filename}')
            frame.imshow(photo, extent=(0, width, height, 0), interpolation='antialiased')
        frame.set(xlim=(-.013 * width, 1.013 * width), ylim=(1.013 * height, -.013 * height), xticks=[], yticks=[])
        frame.set_aspect('equal', adjustable='box')
        for spine in frame.spines.values():
            spine.set_visible(False)

        def box(coords, color, linestyle, tag, prediction=False):
            x1, y1, x2, y2 = coords
            outline = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color,
                                linewidth=1.6, linestyle=linestyle)
            outline.set_path_effects([patheffects.withStroke(linewidth=2.9, foreground='#182023')])
            frame.add_patch(outline)
            if prediction:
                xy, horizontal = (x2, y2), 'right'
                offset = (-3, -4) if y2 < .85 * height else (-3, 3)
            elif x2 - x1 < .12 * width:
                xy, horizontal, offset = (x2, y1), 'left', (5, 4)
            else:
                xy, horizontal = (x1, y1), 'left'
                offset = (3, 4) if y1 > .15 * height else (3, -3)
            frame.annotate(tag, xy, xytext=offset, textcoords='offset points',
                           ha=horizontal, va='bottom' if offset[1] > 0 else 'top',
                           fontsize=8, weight='bold', color=color,
                           bbox={'facecolor': '#122027', 'alpha': .90, 'edgecolor': 'none', 'pad': 1.5})

        reference = [i for i, row in enumerate(case['truth']) if CLASS_NAMES[row[0]] == target_class]
        matches = [row for row in case['predictions'] if row[6] in reference]
        prediction = max(matches, key=lambda row: row[4], default=None)
        status = 'Missed at the selected confidence'
        if reference:
            index_of_box = prediction[6] if prediction else reference[0]
            box(case['truth'][index_of_box][1:5], REFERENCE_COLOR, '-', f'Truth {target_class}')
            if prediction:
                status = f'Matched {target_class} box'
        else:
            prediction = max(case['predictions'], key=lambda row: row[4], default=None)
            status = 'False alert on a negative frame' if prediction else 'No alert on a negative frame'
        if prediction:
            box(prediction[:4], PREDICTION_COLOR, '--', f'{CLASS_NAMES[prediction[5]]} {prediction[4]:.2f}', True)
        axes.text(center, .94, label, ha='center', va='center', fontsize=12, color=INK)
        axes.text(center, .19, status, ha='center', va='center', fontsize=9.5, color=INK)
        axes.text(center, .11, filename, ha='center', va='center', fontsize=8.5, color=SLATE)
    axes.legend(handles=[Line2D([], [], color='#17796b', lw=2, label='Dataset box'),
                         Line2D([], [], color='#be601e', lw=2, ls='--', label='Model box and confidence')],
                loc='lower center', bbox_to_anchor=(.50, -.10), ncol=2, frameon=False, fontsize=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True, help='Images directory of the fixed test split')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='v8n_aug')
    args = parser.parse_args()
    results = json.loads((args.results / 'benchmark.json').read_text(encoding='utf-8'))['experiments']
    cases = json.loads((args.results / f'{args.model}_test_boxes.json').read_text(encoding='utf-8'))
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
    title(matrix_axes, '(c) Held-out confusion')
    selected = results[args.model]
    matrix = np.array(selected['test_confusion_true_by_predicted'])
    operating = selected['test_operating_point']
    if not np.array_equal(matrix.diagonal()[:2], np.asarray(operating['matrix']).diagonal()[:2]):
        raise ValueError('Confusion matrix and reported true-positive counts disagree')
    if not np.array_equal(matrix.sum(1)[:2], list(operating['truth_support'].values())):
        raise ValueError('Confusion matrix and reported test support disagree')
    percentages = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    matrix_axes.imshow(percentages, cmap='Blues', vmin=0, vmax=1)
    tick_labels = ['Smoke', 'Fire', 'Background']
    matrix_axes.set_xticks(range(3), tick_labels, fontsize=11)
    matrix_axes.set_yticks(range(3), tick_labels, fontsize=11)
    matrix_axes.set(xlabel=f"Predicted class\n{LABELS[args.model]} · conf {operating['threshold']:.2f}",
                    ylabel='Reference class')
    for row, col in np.ndindex(matrix.shape):
        color = 'white' if percentages[row, col] > .55 else INK
        count = '–' if (row, col) == (2, 2) else str(matrix[row, col])
        matrix_axes.text(col, row, count, color=color, va='center', ha='center', size=14)

    bottom = figure.add_subplot(grid[2, :])
    title(bottom, '(d) Annotated D-Fire test photographs')
    qualitative(bottom, cases, args.images)
    figure.savefig(args.output, dpi=220, facecolor='white')


if __name__ == '__main__':
    main()
