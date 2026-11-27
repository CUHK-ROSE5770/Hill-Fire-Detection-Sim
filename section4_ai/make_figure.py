"""Draw publication-grade Figure 3 from archived benchmark data, validation trajectories, and test box coordinates."""

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


INK = '#152a36'
SLATE = '#4a6270'
TEAL = '#17796b'     # Unified Recall accent across (b) & (c)
NAVY = '#164e75'     # Unified Precision accent across (b) & (c)
AMBER = '#c67632'    # Unified mAP@0.5 / Small-target accent
CLASS_NAMES = ('smoke', 'fire')
REFERENCE_COLOR = '#69edb6'
PREDICTION_COLOR = '#ffb454'
EXAMPLES = (
    ('AoF07925.jpg', 'Hillside fire', 'fire'),
    ('AoF07976.jpg', 'Smoke plume', 'smoke'),
    ('AoF07986.jpg', 'Small smoke plume', 'smoke'),
    ('AoF06871.jpg', 'Background frame', None),
)
DISPLAY_NAMES = {
    'v8n_base_640': 'v8n 640 px',
    'v8n_aerial_aug': 'v8n aug WiSE-FT',
    'v8s_base_640': 'v8s 640 px',
    'v8s_multiscale_768': 'v8s 768 TTA',
}


def _stage_card(axes, x, y, w, h, tag, heading, subtext, accent, face):
    axes.add_patch(FancyBboxPatch(
        (x + .0025, y - .026), w, h, boxstyle='round,pad=.003,rounding_size=.022',
        facecolor='#d7e1e7', edgecolor='none', alpha=.55, zorder=1,
    ))
    axes.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=.003,rounding_size=.022',
        facecolor=face, edgecolor='#b4c6d0', linewidth=1.0, zorder=2,
    ))
    bar_h = .165
    axes.add_patch(FancyBboxPatch(
        (x + .004, y + h - bar_h - .015), w - .008, bar_h,
        boxstyle='round,pad=.001,rounding_size=.014',
        facecolor=accent, edgecolor='none', zorder=3,
    ))
    axes.text(x + w / 2, y + h - bar_h / 2 - .015, tag,
              ha='center', va='center', size=7.8, color='white', fontweight='bold', zorder=4)
    axes.text(x + w / 2, y + h * .46, heading,
              ha='center', va='center', size=9.8, color=INK, fontweight='bold', zorder=4)
    axes.text(x + w / 2, y + h * .18, subtext,
              ha='center', va='center', size=8.0, color=SLATE, zorder=4)


def pipeline(axes):
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.axis('off')

    axes.add_patch(FancyBboxPatch(
        (.004, .29), .785, .68, boxstyle='round,pad=.004,rounding_size=.022',
        facecolor='#f6fafb', edgecolor='#c5d6df', linestyle='--', linewidth=1.0, zorder=0,
    ))
    axes.text(.014, .938, 'On-board Edge Compute (Jetson Orin Nano · TensorRT FP16)',
              ha='left', va='top', size=8.3, color=NAVY, fontweight='bold', zorder=2)

    axes.add_patch(FancyBboxPatch(
        (.804, .29), .192, .68, boxstyle='round,pad=.004,rounding_size=.022',
        facecolor='#f4f7fb', edgecolor='#bdd0df', linestyle='--', linewidth=1.0, zorder=0,
    ))
    axes.text(.900, .938, 'Swarm & Planning Layer',
              ha='center', va='top', size=8.3, color='#284c72', fontweight='bold', zorder=2)

    cards = [
        (.016, 'STAGE 1 · RGB', 'YOLOv8s Detector', '640 px single-pass', NAVY, '#ffffff'),
        (.212, 'STAGE 2 · GATE', 'Peak Gate & NMS', 'Per-class · IoU 0.50', '#1f6a7d', '#ffffff'),
        (.408, 'STAGE 3 · TRACK', '3-Frame Persistence', 'Spatial IoU ≥ 0.30', TEAL, '#ffffff'),
        (.604, 'STAGE 4 · LWIR', 'Thermal Cross-Check', 'ΔT ≥ 5 °C (≥ 2/3 hits)', '#b85d24', '#fffaf6'),
        (.815, 'STAGE 5 · MESH', 'Ray–DEM & Dispatch', '3D map · APF / EGO', '#2b547e', '#ffffff'),
    ]
    w, h, y = .168, .50, .335
    for idx, (x, tag, heading, subtext, accent, face) in enumerate(cards):
        _stage_card(axes, x, y, w, h, tag, heading, subtext, accent, face)
        if idx < len(cards) - 1:
            next_x = cards[idx + 1][0]
            axes.annotate(
                '', xy=(next_x - .005, y + h * .46), xytext=(x + w + .005, y + h * .46),
                arrowprops={'arrowstyle': '-|>', 'color': SLATE, 'lw': 1.5, 'mutation_scale': 12},
                zorder=5,
            )

    # Candidate fallback branch routed cleanly at y=0.19
    axes.plot([.492, .492, .899, .899], [.325, .19, .19, .32], color='#5c7684', linestyle=':', linewidth=1.3, zorder=2)
    axes.annotate(
        '', xy=(.899, .33), xytext=(.899, .26),
        arrowprops={'arrowstyle': '-|>', 'color': '#5c7684', 'lw': 1.3, 'mutation_scale': 10}, zorder=3,
    )
    axes.text(.245, .145, 'RGB-only persistent cue (no thermal rise) → visual candidate for cross-view / operator review',
              ha='center', va='center', size=8.0, color=SLATE,
              bbox={'facecolor': '#f3f7f9', 'edgecolor': '#ccd9e0', 'boxstyle': 'round,pad=.24'}, zorder=4)

    # Co-registered LWIR input badge raised to y=0.075 to leave generous whitespace above row 2
    axes.annotate(
        'Co-registered LWIR frame', xy=(.688, .33), xytext=(.688, .075),
        ha='center', va='center', fontsize=8.0, color='#9a4919', fontweight='bold',
        bbox={'facecolor': '#fff5ee', 'edgecolor': '#dfb599', 'boxstyle': 'round,pad=.22'},
        arrowprops={'arrowstyle': '-|>', 'color': '#b85d24', 'lw': 1.4, 'mutation_scale': 10},
        zorder=6,
    )


def title(axes, text, pad=10):
    axes.set_title(text, loc='left', color=INK, fontweight='bold', fontsize=12.4, pad=pad)


def plot_curves(axes, results_dir):
    ft_csv = results_dir / 'v8n_aerial_aug_training.csv'
    with ft_csv.open(newline='', encoding='utf-8') as stream:
        records = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(stream)]

    epochs = [r['epoch'] for r in records]
    map50 = [r['metrics/mAP50(B)'] * 100 for r in records]
    prec = [r['metrics/precision(B)'] * 100 for r in records]
    rec = [r['metrics/recall(B)'] * 100 for r in records]

    # Shade close_mosaic phase (Epochs 4-5)
    axes.axvspan(3.5, 5.18, color='#eef5f3', alpha=.85, zorder=0)
    axes.axvline(3.5, color='#8eb2a9', linestyle='--', linewidth=1.0, zorder=1)
    axes.text(4.34, 60.2, 'Mosaic disabled\n(close_mosaic=2)', ha='center', va='bottom',
              fontsize=7.8, color='#2c5f55', fontweight='bold')

    # Unified cross-panel color semantics: Navy = Precision, Teal = Recall, Amber = mAP@0.5
    axes.plot(epochs, map50, label='Val mAP@0.5', color=AMBER, marker='D', linewidth=2.3, markersize=5.5, zorder=4)
    axes.plot(epochs, prec, label='Val Precision', color=NAVY, marker='o', linewidth=2.0, markersize=5.0, zorder=3)
    axes.plot(epochs, rec, label='Val Recall', color=TEAL, marker='s', linewidth=2.0, markersize=5.0, zorder=3)

    peak_idx = int(np.argmax(map50))
    axes.annotate(
        f'Peak {map50[peak_idx]:.1f}%',
        xy=(epochs[peak_idx], map50[peak_idx]),
        xytext=(epochs[peak_idx] - .55, map50[peak_idx] + 3.2),
        fontsize=8.2, color='#9c571d', fontweight='bold',
        arrowprops={'arrowstyle': '->', 'color': AMBER, 'lw': 1.1},
    )

    axes.set(xlabel='Fine-tuning epoch (v8n · 640 px · val split)', ylabel='Validation metric (%)',
             xlim=(0.82, 5.18), ylim=(58, 90))
    axes.set_xticks([1, 2, 3, 4, 5])
    axes.set_yticks([60, 65, 70, 75, 80, 85, 90])
    axes.tick_params(labelsize=9.8)
    axes.xaxis.label.set_size(10.3)
    axes.yaxis.label.set_size(10.5)
    axes.grid(axis='y', alpha=.25)
    axes.spines[['right', 'top']].set_visible(False)
    axes.legend(loc='lower left', frameon=True, facecolor='white', framealpha=.94,
                edgecolor='#d8e3e8', fontsize=8.2)


def plot_industrial_pr(axes, prod_models):
    arms = [
        ('v8n_base_640', 'v8n\n640 px'),
        ('v8n_aerial_aug', 'v8n aug\nWiSE-FT'),
        ('v8s_base_640', 'v8s\n640 px\n(Deployed)'),
        ('v8s_multiscale_768', 'v8s\n768 TTA'),
    ]
    x = np.arange(len(arms))
    w = 0.255
    overall_p = [prod_models[k]['test']['frames']['localized_candidate_precision'] * 100 for k, _ in arms]
    overall_r = [prod_models[k]['test']['frames']['localized_frame_recall'] * 100 for k, _ in arms]
    small_r = [prod_models[k]['test']['boxes']['small_recall']['smoke'] * 100 for k, _ in arms]

    # Subtle highlight behind the deployed on-board model (v8s 640 px)
    axes.axvspan(2 - 0.44, 2 + 0.44, color='#eaf4f1', alpha=.9, zorder=0)

    # Low-zorder reference lines with clean right-margin labels (never touching bar numbers)
    axes.axhline(90, color='#1b695c', linestyle='--', linewidth=1.1, alpha=.55, zorder=1)
    axes.axhline(85, color=NAVY, linestyle=':', linewidth=1.1, alpha=.55, zorder=1)
    axes.text(3.47, 90.0, '90% R', ha='left', va='center', fontsize=7.6, color='#145a4e', fontweight='bold', zorder=4)
    axes.text(3.47, 85.0, '85% P', ha='left', va='center', fontsize=7.6, color=NAVY, fontweight='bold', zorder=4)

    b1 = axes.bar(x - w, overall_p, width=w * 0.90, color=NAVY, label='Frame Prec.', zorder=3)
    b2 = axes.bar(x, overall_r, width=w * 0.90, color=TEAL, label='Frame Rec.', zorder=3)
    b3 = axes.bar(x + w, small_r, width=w * 0.90, color=AMBER, label='Small-Smoke R', zorder=3)

    for bars in (b1, b2, b3):
        for bar in bars:
            val = bar.get_height()
            axes.text(
                bar.get_x() + bar.get_width() / 2, val + 0.65, f'{val:.1f}',
                ha='center', va='bottom', fontsize=7.2, color=INK, fontweight='bold', zorder=6,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.88, 'pad': 0.55},
            )

    axes.set_xticks(x, [lbl for _, lbl in arms], fontsize=9.1)
    # Emphasize the deployed on-board model tick label
    tick_labels = axes.get_xticklabels()
    tick_labels[2].set_fontweight('bold')
    tick_labels[2].set_color('#0f594c')

    axes.set(ylabel='Held-out score (%)', xlim=(-0.54, 3.88), ylim=(68, 101.4))
    axes.set_yticks([70, 75, 80, 85, 90, 95, 100])
    axes.tick_params(axis='y', labelsize=9.8)
    axes.yaxis.label.set_size(10.5)
    axes.grid(axis='y', alpha=.22, zorder=0)
    axes.spines[['right', 'top']].set_visible(False)
    # Clean single-row legend with generous breathing room below the title
    axes.legend(loc='lower center', bbox_to_anchor=(0.48, 1.015), ncol=3, frameon=False,
                fontsize=8.1, columnspacing=1.0, handlelength=1.4, borderaxespad=0.)


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
    axes.set_xticks(range(3), tick_labels, fontsize=9.8)
    axes.set_yticks(range(3), tick_labels, fontsize=9.8)
    f = selected['test']['frames']
    clean_name = DISPLAY_NAMES.get(model_name, model_name)
    axes.set(
        xlabel=f"Predicted ({clean_name})\nFrame P {f['localized_candidate_precision']*100:.1f}% · R {f['localized_frame_recall']*100:.1f}%",
        ylabel='Reference class',
    )
    axes.xaxis.label.set_size(9.8)
    axes.yaxis.label.set_size(10.2)
    for row, col in np.ndindex(matrix.shape):
        color = 'white' if percentages[row, col] > .55 else INK
        count = '–' if (row, col) == (2, 2) else str(matrix[row, col])
        axes.text(col, row, count, color=color, va='center', ha='center', size=12.2, fontweight='bold')


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
        axes.text(center, .94, label, ha='center', va='center', fontsize=11.0, color=INK, fontweight='bold')
        ok_color = '#145a46' if 'Missed' not in status and 'False' not in status else '#9c3618'
        axes.text(center, .19, status, ha='center', va='center', fontsize=9.3, color=ok_color, fontweight='bold')
        axes.text(center, .10, filename, ha='center', va='center', fontsize=8.3, color=SLATE)
    axes.legend(handles=[Line2D([], [], color=TEAL, lw=2, label='Dataset ground-truth box'),
                         Line2D([], [], color='#be601e', lw=2, ls='--', label='Deployed v8s 640 px box and confidence')],
                loc='lower center', bbox_to_anchor=(.50, -.11), ncol=2, frameon=False, fontsize=9.8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True, help='Images directory of the fixed test split')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='v8s_base_640')
    args = parser.parse_args()

    prod_models = json.loads((args.results / 'production_benchmark.json').read_text(encoding='utf-8'))['models']
    cases = json.loads((args.results / f'{args.model}_test_boxes.json').read_text(encoding='utf-8'))

    figure = plt.figure(figsize=(13.8, 8.6), layout='constrained', facecolor='white')
    figure.get_layout_engine().set(h_pad=0.14, hspace=0.09, w_pad=0.08)
    grid = figure.add_gridspec(3, 3, height_ratios=[1.02, 1.54, 1.38], width_ratios=[1.16, 1.38, .92])

    design = figure.add_subplot(grid[0, :])
    title(design, '(a) On-board edge AI perception and dual-sensor alert confirmation architecture', pad=6)
    pipeline(design)

    curves = figure.add_subplot(grid[1, 0])
    title(curves, '(b) YOLOv8n aerial fine-tuning trajectories', pad=22)
    plot_curves(curves, args.results)

    bars = figure.add_subplot(grid[1, 1])
    title(bars, '(c) Held-out detection benchmark (600 images)', pad=22)
    plot_industrial_pr(bars, prod_models)

    matrix_axes = figure.add_subplot(grid[1, 2])
    title(matrix_axes, '(d) Held-out box confusion', pad=22)
    plot_confusion(matrix_axes, prod_models, args.model)

    bottom = figure.add_subplot(grid[2, :])
    title(bottom, '(e) Annotated D-Fire held-out test photographs (deployed v8s 640 px)', pad=6)
    qualitative(bottom, cases, args.images)

    figure.savefig(args.output, dpi=240, facecolor='white')


if __name__ == '__main__':
    main()
