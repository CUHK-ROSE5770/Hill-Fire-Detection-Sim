"""Measure held-out AP, calibrate RGB confidence on validation, and export trial models."""

import argparse
import csv
import hashlib
import json
import platform
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils import torch_utils

from prepare_data import CLASSES, IMAGE_SUFFIXES
from train import EXPERIMENTS, dataset_config


CONFIDENCE_GRID = (.001, .003, .005, .01, .02, .03, .05, .075, .10, .15, .20, .25, .35, .50, .65, .80, .95)


def read_truth(image, width, height, imgsz):
    label = image.parent.parent / 'labels' / f'{image.stem}.txt'
    truth = []
    gain = min(imgsz / width, imgsz / height)
    for line in label.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        cls, cx, cy, w, h = map(float, line.split())
        truth.append((int(cls), (cx - w / 2) * width, (cy - h / 2) * height,
                      (cx + w / 2) * width, (cy + h / 2) * height,
                      w * width * h * height * gain ** 2 < 32 ** 2))
    return truth


def overlap(box, boxes):
    intersection = np.maximum(0, np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2])).prod(1)
    union = (box[2:] - box[:2]).prod() + (boxes[:, 2:] - boxes[:, :2]).prod(1) - intersection
    return intersection / np.maximum(union, 1e-9)


def predict_split(model, root, split, imgsz, device, *, conf=.001, iou=.70, max_det=100, augment=False):
    paths = sorted(p for p in (root / split / 'images').iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    rows = []
    results = model.predict(source=str(root / split / 'images'), stream=True, device=device, imgsz=imgsz,
                            conf=conf, iou=iou, max_det=max_det, batch=1, verbose=False, rect=False, augment=augment)
    for result in results:
        image = Path(result.path)
        h, w = result.orig_shape
        truth = read_truth(image, w, h, imgsz)
        preds = sorted(result.boxes.data.cpu().tolist(), key=lambda row: -row[4])
        rows.append((image.name, w, h, truth, preds, result.speed))
    if len(rows) != len(paths):
        raise RuntimeError(f'Prediction count mismatch for {split}: {len(rows)} vs {len(paths)}')
    return rows


def score(rows, conf):
    thresholds = (conf, conf) if isinstance(conf, (int, float)) else conf
    matrix = np.zeros((3, 3), dtype=int)  # rows = true class, columns = predicted class
    small = np.zeros(2, dtype=int)
    small_hit = np.zeros(2, dtype=int)
    negative = false_alarm = 0
    cases = []
    for filename, width, height, truth, preds, _ in rows:
        active = [pred for pred in preds if pred[4] >= thresholds[int(pred[5])]]
        boxes = np.asarray([row[1:5] for row in truth], dtype=float).reshape(-1, 4)
        remaining = np.ones(len(truth), dtype=bool)
        true_classes = np.asarray([row[0] for row in truth], dtype=int)
        for cls, *_, tiny in truth:
            small[cls] += tiny
        if not truth:
            negative += 1
            false_alarm += bool(active)
        matched = []
        for x1, y1, x2, y2, confidence, predicted in active:
            predicted = int(predicted)
            eligible = np.flatnonzero(remaining & (true_classes == predicted))
            similarities = overlap(np.asarray((x1, y1, x2, y2)), boxes[eligible])
            target = -1
            if similarities.size and similarities.max() >= .5:
                target = eligible[similarities.argmax()]
                remaining[target] = False
                cls, *_, tiny = truth[target]
                small_hit[cls] += tiny and cls == predicted
                matrix[cls, predicted] += 1
            else:
                matrix[2, predicted] += 1
            matched.append([x1, y1, x2, y2, confidence, predicted, int(target)])
        for index in np.flatnonzero(remaining):
            matrix[truth[index][0], 2] += 1
        cases.append({'filename': filename, 'width': width, 'height': height,
                      'truth': truth, 'predictions': matched})
    diagonal = matrix.diagonal()[:2]
    precision = diagonal / np.maximum(matrix.sum(0)[:2], 1)
    recall = diagonal / np.maximum(matrix.sum(1)[:2], 1)
    return {
        'threshold': conf,
        'precision': dict(zip(CLASSES, precision.tolist())),
        'recall': dict(zip(CLASSES, recall.tolist())),
        'small_recall': dict(zip(CLASSES, (small_hit / np.maximum(small, 1)).tolist())),
        'small_support': dict(zip(CLASSES, small.tolist())),
        'truth_support': dict(zip(CLASSES, matrix.sum(1)[:2].tolist())),
        'negative_images': negative,
        'negative_images_with_prediction': false_alarm,
        'negative_image_alert_rate': false_alarm / max(negative, 1),
        'matrix': matrix.tolist(),
    }, cases


def confusion_matrix(matrix, cases):
    """Pair only residual wrong-class predictions, preserving the operating-point true positives."""
    matrix = np.array(matrix, dtype=int, copy=True)
    for case in cases:
        truth, predictions = case['truth'], case['predictions']
        occupied = {pred[6] for pred in predictions if pred[6] >= 0}
        for x1, y1, x2, y2, _, predicted, match in predictions:
            if match >= 0:
                continue
            eligible = [j for j, item in enumerate(truth) if j not in occupied and item[0] != predicted]
            if not eligible:
                continue
            ious = overlap(np.array([x1, y1, x2, y2]), np.array([truth[j][1:5] for j in eligible]))
            best = ious.argmax()
            if ious[best] >= .5:
                index = eligible[best]
                cls = truth[index][0]
                matrix[cls, 2] -= 1
                matrix[2, predicted] -= 1
                matrix[cls, predicted] += 1
                occupied.add(index)
    return matrix.tolist()


def frame_candidate_metrics(cases):
    """Count images with at least one same-class IoU match, separately from box recall."""
    positives = sum(bool(case['truth']) for case in cases)
    candidate_frames = sum(bool(case['predictions']) for case in cases)
    localized_frames = sum(any(prediction[6] >= 0 for prediction in case['predictions']) for case in cases)
    return {'positive_frames': positives, 'candidate_frames': candidate_frames, 'localized_frames': localized_frames,
            'localized_frame_recall': localized_frames / max(1, positives),
            'localized_candidate_precision': localized_frames / max(1, candidate_frames)}


def gate_frames(rows, thresholds, peak_threshold):
    """Suppress all RGB proposals when a frame has no sufficiently confident surviving box."""
    for filename, width, height, labels, boxes, timing in rows:
        accepted = [box for box in boxes if box[4] >= thresholds[int(box[5])]]
        if max((box[4] for box in accepted), default=0.) < peak_threshold:
            accepted = []
        yield (filename, width, height, labels, accepted, timing)


def calibrate(rows):
    curve = [score(rows, value)[0] for value in CONFIDENCE_GRID]
    feasible = [entry for entry in curve if min(entry['recall'].values()) > .90]
    if feasible:
        selected = max(feasible, key=lambda row: (sum(row['precision'].values()), row['threshold']))
    else:
        def macro_f2(entry):
            values = []
            for cls in CLASSES:
                p, r = entry['precision'][cls], entry['recall'][cls]
                values.append(5 * p * r / max(4 * p + r, 1e-9))
            return sum(values) / len(values)

        selected = max(curve, key=lambda row: (macro_f2(row), row['threshold']))
    return selected, bool(feasible), curve


def package_checkpoint(source, output, name):
    """Ship portable metadata alongside the unchanged parameters of our local checkpoint."""
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    paths = {'model': EXPERIMENTS[name][0], 'data': 'data.yaml', 'project': 'runs', 'save_dir': f'runs/{name}'}
    checkpoint['train_args'].update(paths)
    checkpoint['model'].args.update(paths)
    checkpoint['model'].criterion = None
    torch.save(checkpoint, output)


def artifact_digest(path):
    with path.open('rb') as file:
        return {'sha256': hashlib.file_digest(file, 'sha256').hexdigest(), 'bytes': path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--only', nargs='+', choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--export', nargs='+', choices=EXPERIMENTS, default=['v8n_base', 'v8n_aug'])
    args = parser.parse_args()
    torch_utils.NUM_THREADS = args.threads
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    args.results.mkdir(parents=True, exist_ok=True)
    args.models.mkdir(parents=True, exist_ok=True)
    source = args.data.resolve().parent
    data = dataset_config(args.data, args.runs)
    measurements = {}
    for name in args.only:
        run = args.runs / name
        checkpoint = run / 'weights' / 'best.pt'
        model = YOLO(checkpoint)
        training_args = model.ckpt['train_args']
        imgsz = int(training_args['imgsz'])
        params = sum(p.numel() for p in model.model.parameters())
        package_checkpoint(checkpoint, args.models / f'{name}.pt', name)
        shutil.copyfile(run / 'results.csv', args.results / f'{name}_training.csv')
        val_rows = predict_split(model, source, 'val', imgsz, args.device)
        operating, feasible, curve = calibrate(val_rows)
        test_rows = predict_split(model, source, 'test', imgsz, args.device)
        threshold = operating['threshold']
        test, cases = score(test_rows, threshold)
        shared, _ = score(test_rows, .10)
        confusion = confusion_matrix(test['matrix'], cases)
        latency = np.asarray([sum(item[-1].values()) for item in test_rows[10:]])
        report = model.val(data=str(data), split='test', imgsz=imgsz, batch=1, device=args.device,
                           workers=0, plots=False, conf=.001, iou=.7, max_det=100, rect=False,
                           project=str(args.runs.resolve()), name=f'{name}_test', verbose=False)
        measurements[name] = {
            'parameters': params,
            'map50': float(report.box.map50),
            'map50_95': float(report.box.map),
            'validation_target_met': feasible,
            'validation_operating_point': operating,
            'validation_curve': curve,
            'test_operating_point': test,
            'test_shared_threshold': shared,
            'test_confusion_true_by_predicted': confusion,
            'test_images': len(test_rows),
            'pipeline_mean_ms': float(latency.mean()),
            'pipeline_p95_ms': float(np.percentile(latency, 95)),
            'pipeline_fps': float(1000 / latency.mean()),
            'imgsz': imgsz,
            'training': {key: training_args[key] for key in (
                'epochs', 'batch', 'seed', 'freeze', 'optimizer', 'lr0', 'mosaic', 'scale', 'flipud'
            )},
        }
        if name in args.export:
            packaged = YOLO(args.models / f'{name}.pt')
            packaged.export(format='onnx', imgsz=imgsz, opset=17, dynamic=False, simplify=False,
                            device='cpu', batch=1, half=False, nms=False)
        (args.results / f'{name}_test_boxes.json').write_text(json.dumps(cases, separators=(',', ':')) + '\n',
                                                             encoding='utf-8')

    output = {'platform': platform.platform(), 'processor': platform.processor(), 'device': args.device,
              'torch': torch.__version__, 'ultralytics': ultralytics.__version__, 'threads': args.threads,
              'experiments': measurements,
              'artifacts': {path.name: artifact_digest(path) for path in sorted(args.models.iterdir())
                            if path.suffix in {'.pt', '.onnx'}}}
    (args.results / 'benchmark.json').write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    with (args.results / 'benchmark.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(('experiment', 'parameters', 'map50', 'map50_95', 'threshold', 'smoke_recall',
                         'fire_recall', 'small_smoke_recall', 'negative_frame_rate', 'rgb_pipeline_fps'))
        for name, row in measurements.items():
            test = row['test_operating_point']
            writer.writerow((name, row['parameters'], row['map50'], row['map50_95'], test['threshold'],
                             test['recall']['smoke'], test['recall']['fire'], test['small_recall']['smoke'],
                             test['negative_image_alert_rate'], row['pipeline_fps']))


if __name__ == '__main__':
    main()
