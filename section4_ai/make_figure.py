"""Draw publication-grade Figure 3 from archived benchmark data, training curves, and test box coordinates."""

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
    labels = (
        'RGB YOLOv8\n(640 px FP16)',
        'Class Peak Gate\n+ IoU 0.50 NMS',
        '3-Frame Track\nPersistence',
        'Aligned LWIR\nΔT ≥ 5 °C Check',
        'Ray–DEM & Mesh\nSwarm Alert',
    )
    for index, label in enumerate(labels):
        x = .008 + index * .200
        axes.add_patch(FancyBboxPatch(
            (x, .27), .178, .42, boxstyle='round,pad=.004,rounding_size=.018',
            facecolor='#e7f4f0' if index < 4 else '#ebf2f7', edgecolor='#9ab3bf', linewidth=1.1,
        ))
        axes.text(x + .089, .48, label, ha='center', va='center', size=11.5, color=INK, fontweight='semibold')
        if index < len(labels) - 1:
            axes.annotate('', (x + .198, .48), (x + .180, .48),
                          arrowprops={'arrowstyle': '->', 'color': SLATE, 'lw': 1.5})
    axes.text(.008, .06,
              'On-board confirmation: persistent RGB cues without thermal rise remain visual candidates for cross-view or operator verification.',
              color=SLATE, size=10.5)


def title(axes, text):
    axes.set_title(text, loc='left', color=INK, fontweight='bold', fontsize=13.5, pad=8)


def plot_curves(axes, results_dir, prod_models):
    local_specs = [
        ('v8n_base', 'v8n 256 px (5 ep)', '#7d98a8', 'o', '--'),
        ('v8s_base', 'v8s 256 px (5 ep)', '#c27b57', 's', '--'),
        ('v8n_aug', 'v8n aug 256 px (5 ep)', '#72a89c', '^', ':'),
    ]
    for name, label, color, marker, ls in local_specs:
        csv_path = results_dir / f'{name}_training.csv'
        if csv_path.exists():
            with csv_path.open(newline='', encoding='utf-8') as stream:
                records = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
            axes.plot([r['epoch'] for r in records], [r['metrics/mAP50(B)'] * 100 for r in records],
                      label=label, color=color, marker=marker, linestyle=ls, linewidth=1.7, markersize=4.5)

    ft_csv = results_dir / 'v8n_aerial_aug_training.csv'
    if ft_csv.exists():
        with ft_csv.open(newline='', encoding='utf-8') as stream:
            records = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
        axes.plot([r['epoch'] for r in records], [r['metrics/mAP50(B)'] * 100 for r in records],
                  label='v8n aerial FT 640 px (val)', color='#17796b', marker='D', linewidth=2.4, markersize=5.5)

    if 'v8s_base_640' in prod_models:
        s_map = prod_models['v8s_base_640']['standard_ap']['map50'] * 100
        axes.axhline(s_map, color='#a44c23', linestyle='-.', linewidth=1.7, label=f'v8s 640 px test ({s_map:.1f}%)')
    if 'v8n_base_640' in prod_models:
        n_map = prod_models['v8n_base_640']['standard_ap']['map50'] * 100
        axes.axhline(n_map, color='#164e75', linestyle=':', linewidth=1.7, label=f'v8n 640 px test ({n_map:.1f}%)')

    axes.set(xlabel='Epoch', ylabel='mAP@0.5 (%)', ylim=(8, 92))
    axes.set_xticks([1, 2, 3, 4, 5])
    axes.tick_params(labelsize=10)
    axes.xaxis.label.set_size(11)
    axes.yaxis.label.set_size(11)
    axes.grid(axis='y', alpha=.25)
    axes.spines[['right', 'top']].set_visible(False)
    axes.legend(loc='center left', bbox_to_anchor=(.01, .62), ncol=2, frameon=True,
                facecolor='white', framealpha=.92, edgecolor='none', fontsize=7.4, columnspacing=.8, handlelength=1.8)


def plot_industrial_pr(axes, prod_models):
    arms = [
        ('v8n_base_640', 'v8n\n640 px'),
        ('v8n_aerial_aug', 'v8n aug\nWiSE-FT'),
        ('v8s_base_640', 'v8s\n640 px'),
        ('v8s_multiscale_768', 'v8s\n768 TTA'),
    ]
    x = np.arange(len(arms))
    w = 0.26
    overall_p = [prod_models[k]['test']['frames']['localized_candidate_precision'] * 100 for k, _ in arms]
    overall_r = [prod_models[k]['test']['frames']['localized_frame_recall'] * 100 for k, _ in arms]
    small_r = [prod_models[k]['test']['boxes']['small_recall']['smoke'] * 100 for k, _ in arms]

    b1 = axes.bar(x - w, overall_p, width=w * 0.92, color='#164e75', label='Frame Prec.')
    b2 = axes.bar(x, overall_r, width=w * 0.92, color='#17796b', label='Frame Rec.')
    b3 = axes.bar(x + w, small_r, width=w * 0.92, color='#c67632', label='Small-Smoke R')
    axes.axhline(90, color='#0f594c', linestyle='--', linewidth=1.2, alpha=.85, label='90% Recall')
    axes.axhline(85, color='#164e75', linestyle=':', linewidth=1.2, alpha=.85, label='85% Prec.')

    for bars in (b1, b2, b3):
        for bar in bars:
            val = bar.get_height()
            axes.text(bar.get_x() + bar.get_width() / 2, val + 0.7, f'{val:.1f}',
                      ha='center', va='bottom', fontsize=7.2, color=INK, fontweight='bold')

    axes.set_xticks(x, [lbl for _, lbl in arms], fontsize=9.5)
    axes.set(ylabel='Held-out score (%)', ylim=(68, 108))
    axes.tick_params(axis='y', labelsize=10)
    axes.yaxis.label.set_size(11)
    axes.grid(axis='y', alpha=.25)
    axes.spines[['right', 'top']].set_visible(False)
    axes.legend(loc='upper center', ncol=3, frameon=True, facecolor='white', framealpha=.92,
                edgecolor='none', fontsize=7.4, columnspacing=.8, handlelength=1.6)


def plot_confusion(axes, prod_models, model_name):
    selected = prod_models[model_name]
    matrix = np.array(selected['test']['confusion_true_by_predicted'])
    operating = selected['test']['boxes']
    if not np.array_equal(matrix.diagonal()[:2], np.asarray(operating['matrix']).diagonal()[:2]):
        raise ValueError('Confusion matrix and reported true-positive counts disagree')
    if not np.array_equal(matrix.sum(1)[:2], list(operating['truth_support'].values())):
        raise ValueError('Confusion matrix and reported test support disagree')
    percentages = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    axes.imshow(percentages, cmap='Blues', vmin=0, vmax=1)
    tick_labels = ['Smoke', 'Fire', 'Bg']
    axes.set_xticks(range(3), tick_labels, fontsize=10)
    axes.set_yticks(range(3), tick_labels, fontsize=10)
    f = selected['test']['frames']
    axes.set(
        xlabel=f"Predicted ({model_name})\nFrame P {f['localized_candidate_precision']*100:.1f}% · R {f['localized_frame_recall']*100:.1f}%",
        ylabel='Reference class',
    )
    axes.xaxis.label.set_size(10)
    axes.yaxis.label.set_size(10.5)
    for row, col in np.ndindex(matrix.shape):
        color = 'white' if percentages[row, col] > .55 else INK
        count = '–' if (row, col) == (2, 2) else str(matrix[row, col])
        axes.text(col, row, count, color=color, va='center', ha='center', size=12.5, fontweight='bold')


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
                                linewidth=1.7, linestyle=linestyle)
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
                status = f'Matched {target_class} ({prediction[4]:.2f})'
        else:
            prediction = max(case['predictions'], key=lambda row: row[4], default=None)
            status = 'False alert on a negative frame' if prediction else 'Clean negative frame (no alert)'
        if prediction:
            box(prediction[:4], PREDICTION_COLOR, '--', f'{CLASS_NAMES[prediction[5]]} {prediction[4]:.2f}', True)
        axes.text(center, .94, label, ha='center', va='center', fontsize=11.5, color=INK, fontweight='bold')
        axes.text(center, .19, status, ha='center', va='center', fontsize=9.5, color='#145a46' if 'Missed' not in status and 'False' not in status else '#9c3618', fontweight='semibold')
        axes.text(center, .10, filename, ha='center', va='center', fontsize=8.5, color=SLATE)
    axes.legend(handles=[Line2D([], [], color='#17796b', lw=2, label='Dataset ground-truth box'),
                         Line2D([], [], color='#be601e', lw=2, ls='--', label='Calibrated YOLOv8s (640 px) box and confidence')],
                loc='lower center', bbox_to_anchor=(.50, -.11), ncol=2, frameon=False, fontsize=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True, help='Images directory of the fixed test split')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='v8s_base_640')
    args = parser.parse_args()

    prod_models = json.loads((args.results / 'production_benchmark.json').read_text(encoding='utf-8'))['models']
    cases = json.loads((args.results / f'{args.model}_test_boxes.json').read_text(encoding='utf-8'))

    figure = plt.figure(figsize=(13.6, 7.8), layout='constrained', facecolor='white')
    grid = figure.add_gridspec(3, 3, height_ratios=[.66, 1.52, 1.38], width_ratios=[1.18, 1.32, .95])

    design = figure.add_subplot(grid[0, :])
    title(design, '(a) On-board edge AI detection and alert confirmation pipeline')
    pipeline(design)

    curves = figure.add_subplot(grid[1, 0])
    title(curves, '(b) Learning & convergence curves')
    plot_curves(curves, args.results, prod_models)

    bars = figure.add_subplot(grid[1, 1])
    title(bars, '(c) Held-out P/R vs. industrial targets')
    plot_industrial_pr(bars, prod_models)

    matrix_axes = figure.add_subplot(grid[1, 2])
    title(matrix_axes, '(d) Held-out box confusion')
    plot_confusion(matrix_axes, prod_models, args.model)

    bottom = figure.add_subplot(grid[2, :])
    title(bottom, '(e) Annotated D-Fire held-out test photographs (calibrated YOLOv8s @ 640 px)')
    qualitative(bottom, cases, args.images)

    figure.savefig(args.output, dpi=220, facecolor='white')


if __name__ == '__main__':
    main()
