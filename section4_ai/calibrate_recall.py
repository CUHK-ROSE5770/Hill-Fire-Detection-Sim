"""Calibrate a published D-Fire checkpoint on disjoint images and score the fixed test split."""

import argparse
import csv
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils import torch_utils

from evaluate import artifact_digest, predict_split, score
from prepare_data import CLASSES, read_index


INPUT_SIZE = 640
PREDICTION_SETTINGS = {'conf': .0003, 'iou': .80, 'max_det': 300}
GRID = (.0003, .001, .002, .003, .005, .0075, .01, .015, .02, .03, .05, .075, .10, .15, .20, .30, .40)
CALIBRATION_MINIMUM = {'smoke': .93, 'fire': .91}


def read_manifest(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def verify_fixed_test(root, rows):
    images, labels = root / 'test/images', root / 'test/labels'
    if {path.name for path in images.iterdir()} != {row['filename'] for row in rows}:
        raise ValueError('The test directory must exactly match the fixed split manifest')
    for row in rows:
        filename = Path(row['filename'])
        if filename.name != row['filename']:
            raise ValueError(f'Invalid test filename: {filename}')
        image_hash = artifact_digest(images / filename)['sha256']
        label_hash = artifact_digest((labels / filename).with_suffix('.txt'))['sha256']
        if image_hash != row['image_sha256'] or label_hash != row['label_sha256']:
            raise ValueError(f'Test bytes do not match the manifest for {filename}')


def build_calibration(archive, destination, calibration, fixed_test):
    fixed_names = {row['filename'] for row in fixed_test}
    fixed_hashes = {row['image_sha256'] for row in fixed_test}
    names = {row['filename'] for row in calibration}
    hashes = {row['image_sha256'] for row in calibration}
    if names & fixed_names or hashes & fixed_hashes or len(names) != len(calibration) or len(hashes) != len(calibration):
        raise ValueError('Calibration images must be distinct from each other and the fixed test split')
    for kind in ('images', 'labels'):
        (destination / kind).mkdir(parents=True)
    with ExitStack() as stack:
        index = read_index([archive], stack)
        for row in calibration:
            filename = Path(row['filename'])
            if filename.name != row['filename']:
                raise ValueError(f'Invalid calibration filename: {filename}')
            record = index['test', filename.stem]
            image = record['images'][1]()
            label = record['labels'][1]()
            if hashlib.sha256(image).hexdigest() != row['image_sha256'] or (
                hashlib.sha256(label).hexdigest() != row['label_sha256']
            ):
                raise ValueError(f'Calibration bytes do not match the manifest for {filename}')
            (destination / 'images' / filename).write_bytes(image)
            (destination / 'labels' / filename.with_suffix('.txt')).write_bytes(label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path, help='Official D-Fire test.tar')
    parser.add_argument('--data', required=True, type=Path, help='Root of the fixed 4,500-image YOLO dataset')
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New evaluation directory')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args()
    package_manifest = Path(__file__).with_name('data')
    fixed_test = [row for row in read_manifest(package_manifest / 'split_manifest.csv') if row['split'] == 'test']
    calibration = read_manifest(package_manifest / 'recall_calibration_manifest.csv')
    verify_fixed_test(args.data, fixed_test)
    args.output.mkdir(parents=True, exist_ok=False)
    build_calibration(args.archive, args.output / 'calibration', calibration, fixed_test)
    torch_utils.NUM_THREADS = args.threads
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    model = YOLO(args.weights)
    if model.names != dict(enumerate(CLASSES)):
        raise ValueError(f'Unexpected class IDs for the published checkpoint: {model.names}')
    calibration_rows = predict_split(model, args.output.resolve(), 'calibration', INPUT_SIZE, args.device,
                                     **PREDICTION_SETTINGS)
    curve = [score(calibration_rows, threshold)[0] for threshold in GRID]
    thresholds = []
    for label in CLASSES:
        feasible = [entry for entry in curve if entry['recall'][label] >= CALIBRATION_MINIMUM[label]]
        if not feasible:
            raise ValueError(f'{label} did not meet the calibration recall requirement')
        thresholds.append(max(feasible, key=lambda row: (row['precision'][label], row['threshold']))['threshold'])
    selected = score(calibration_rows, thresholds)[0]
    test_rows = predict_split(model, args.data.resolve(), 'test', INPUT_SIZE, args.device, **PREDICTION_SETTINGS)
    evidence, _ = score(test_rows, thresholds)
    report = {'checkpoint': artifact_digest(args.weights), 'imgsz': INPUT_SIZE,
              'prediction': PREDICTION_SETTINGS, 'calibration_minimum_recall': CALIBRATION_MINIMUM,
              'thresholds': dict(zip(CLASSES, thresholds)), 'calibration_images': len(calibration_rows),
              'calibration_curve': curve, 'calibration_operating_point': selected,
              'test_images': len(test_rows), 'test_operating_point': evidence}
    (args.output / 'recall_benchmark.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
