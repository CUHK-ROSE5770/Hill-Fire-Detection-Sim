"""Evaluate a fixed, class-aware dual-scale profile using the same YOLOv8s checkpoint."""

import argparse
import json
from pathlib import Path

import cv2
import torch
from torchvision.ops import nms
from ultralytics import YOLO
from ultralytics.utils import torch_utils

from calibrate_recall import build_calibration, read_manifest, verify_fixed_test
from evaluate import artifact_digest, frame_candidate_metrics, predict_split, score


# Fixed on the 300-image calibration set before evaluation on the 600-image holdout.
THRESHOLDS = (.030, .010)
POST_NMS_IOU = .60
POST_MAX_PER_CLASS = 100
BASE_SETTINGS = {'conf': .0003, 'iou': .80, 'max_det': 300}


def merge_multiscale(primary, secondary):
    """Retain primary smoke detections and merge fire detections across both views."""
    merged = []
    for cls, threshold in enumerate(THRESHOLDS):
        sources = primary if cls == 0 else [*primary, *secondary]
        boxes = [box for box in sources if int(box[5]) == cls and box[4] >= threshold]
        if boxes:
            tensor = torch.tensor(boxes, dtype=torch.float32)
            keep = nms(tensor[:, :4], tensor[:, 4], POST_NMS_IOU)[:POST_MAX_PER_CLASS]
            merged.extend(boxes[index] for index in keep.tolist())
    return sorted(merged, key=lambda box: -box[4])


def measure(model, root, split, device):
    main = predict_split(model, root, split, 640, device, **BASE_SETTINGS)
    fire = predict_split(model, root, split, 768, device, augment=True, **BASE_SETTINGS)
    if [row[:3] for row in main] != [row[:3] for row in fire]:
        raise ValueError('The base and wide-view prediction order differs')
    rows = [(*a[:4], merge_multiscale(a[4], b[4]), a[5]) for a, b in zip(main, fire)]
    boxes, cases = score(rows, THRESHOLDS)
    return {'images': len(rows), 'boxes': boxes, 'frames': frame_candidate_metrics(cases)}


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
    print('Evaluating frozen settings on calibration', flush=True)
    dev = measure(model, args.output.resolve(), 'calibration', args.device)
    print('Evaluating the same settings on the held-out split', flush=True)
    held = measure(model, args.data.resolve(), 'test', args.device)
    result = {'weights': artifact_digest(args.weights), 'detector_settings': BASE_SETTINGS,
              'views': [{'imgsz': 640, 'tta': False, 'classes': ['smoke', 'fire']},
                        {'imgsz': 768, 'tta': True, 'classes': ['fire']}],
              'thresholds': {'smoke': THRESHOLDS[0], 'fire': THRESHOLDS[1]}, 'post_nms_iou': POST_NMS_IOU,
              'post_max_per_class': POST_MAX_PER_CLASS, 'calibration': dev, 'test': held}
    (args.output / 'multiscale_benchmark.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(held, indent=2))


if __name__ == '__main__':
    main()
