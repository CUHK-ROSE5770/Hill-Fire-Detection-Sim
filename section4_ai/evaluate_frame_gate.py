"""Compare frozen single-frame candidate gating with the published recall profile."""

import argparse
import json
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils import torch_utils

from calibrate_recall import PREDICTION_SETTINGS, build_calibration, read_manifest, verify_fixed_test
from evaluate import artifact_digest, frame_candidate_metrics, gate_frames, predict_split, score


REFERENCE_THRESHOLDS = (.030, .002)
PEAK_GRID = (.005, .010, .020, .030, .050, .075, .10, .125, .15, .20, .25, .30)


def evaluate_gate(rows, peak):
    evidence, cases = score(gate_frames(rows, REFERENCE_THRESHOLDS, peak), REFERENCE_THRESHOLDS)
    return {'boxes': evidence, 'frames': frame_candidate_metrics(cases)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=6)
    args = parser.parse_args()
    manifests = Path(__file__).with_name('data')
    test = [row for row in read_manifest(manifests / 'split_manifest.csv') if row['split'] == 'test']
    verify_fixed_test(args.data, test)
    calibration = read_manifest(manifests / 'recall_calibration_manifest.csv')
    args.output.mkdir(parents=True, exist_ok=False)
    build_calibration(args.archive, args.output / 'calibration', calibration, test)
    torch_utils.NUM_THREADS = args.threads
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    model = YOLO(args.weights)
    if model.names != {0: 'smoke', 1: 'fire'}:
        raise ValueError(f'Unexpected classes {model.names}')
    print('Choosing the frame gate on calibration', flush=True)
    dev = predict_split(model, args.output.resolve(), 'calibration', 640, args.device, **PREDICTION_SETTINGS)
    options = [(peak, evaluate_gate(dev, peak)) for peak in PEAK_GRID]
    feasible = [(peak, report) for peak, report in options
                if report['frames']['localized_frame_recall'] >= .95 and min(report['boxes']['recall'].values()) > .90]
    peak, calibration_result = max(feasible, default=(0., evaluate_gate(dev, 0.)),
                                   key=lambda row: (row[1]['frames']['localized_candidate_precision'], -row[0]))
    print(f'Frozen frame confidence {peak:g}; evaluating the holdout', flush=True)
    test_rows = predict_split(model, args.data.resolve(), 'test', 640, args.device, **PREDICTION_SETTINGS)
    evidence = {'weights': artifact_digest(args.weights), 'imgsz': 640, 'detector': PREDICTION_SETTINGS,
                'thresholds': {'smoke': REFERENCE_THRESHOLDS[0], 'fire': REFERENCE_THRESHOLDS[1]},
                'frame_peak_confidence': peak, 'calibration_images': len(dev),
                'test_images': len(test_rows), 'calibration': calibration_result,
                'test_without_frame_gate': evaluate_gate(test_rows, 0.), 'test': evaluate_gate(test_rows, peak)}
    (args.output / 'frame_gate_benchmark.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(evidence['test'], indent=2))


if __name__ == '__main__':
    main()
