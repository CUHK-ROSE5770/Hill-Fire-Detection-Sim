"""Measure batch-one RGB processing wall time on the device that runs this script."""

import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import torch_utils

from evaluate import artifact_digest
from prepare_data import IMAGE_SUFFIXES


def summarize(values):
    return {'mean': round(float(np.mean(values)), 3), 'p50': round(float(np.percentile(values, 50)), 3),
            'p95': round(float(np.percentile(values, 95)), 3), 'p99': round(float(np.percentile(values, 99)), 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True, type=Path, help='Ultralytics .pt, .onnx or TensorRT .engine')
    parser.add_argument('--images', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--device-label', default='', help='Board and power mode label for this physical run')
    parser.add_argument('--imgsz', type=int, default=256)
    parser.add_argument('--frames', type=int, default=180)
    parser.add_argument('--warmup', type=int, default=25)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--conf', type=float, default=.10)
    parser.add_argument('--iou', type=float, default=.70)
    parser.add_argument('--max-det', type=int, default=100)
    args = parser.parse_args()
    images = sorted(path for path in args.images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not 1 <= args.frames <= len(images) or args.warmup < 0:
        parser.error('Select a positive measurement frame count within the input directory and nonnegative warm-up')
    torch_utils.NUM_THREADS = args.threads
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    model = YOLO(args.weights, task='detect')
    order = [images[index] for index in np.linspace(0, len(images) - 1, args.frames, dtype=int)]
    warmup = [order[index % len(order)] for index in range(args.warmup)]
    wall_ms = []
    stage_ms = {'preprocess': [], 'inference': [], 'postprocess': []}
    for index, path in enumerate(warmup + order):
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Unable to decode {path}')
        start = time.perf_counter()
        result = model.predict(source=image, device=args.device, imgsz=args.imgsz, rect=False, conf=args.conf,
                               iou=args.iou, max_det=args.max_det, verbose=False, batch=1)[0]
        elapsed = 1000 * (time.perf_counter() - start)
        if index >= len(warmup):
            wall_ms.append(elapsed)
            for name, series in stage_ms.items():
                series.append(result.speed[name])
    result = {'evidence_type': 'physical-device RGB replay', 'limits': 'Excludes file decode, live camera, LWIR and Mesh',
              'platform': platform.platform(), 'processor': platform.processor(), 'device': args.device,
              'device_label': args.device_label, 'torch': torch.__version__, 'weights': args.weights.name,
              'weights_digest': artifact_digest(args.weights), 'torch_threads': args.threads, 'imgsz': args.imgsz,
              'confidence_floor': args.conf, 'nms_iou': args.iou, 'max_det': args.max_det,
              'warmup_frames': args.warmup, 'inputs': [item.name for item in order], 'wall_ms': summarize(wall_ms),
              'stages_ms': {name: summarize(series) for name, series in stage_ms.items()},
              'processing_fps': round(1000 / np.mean(wall_ms), 2)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f"{args.weights.name}: {result['processing_fps']} RGB FPS, p95 {result['wall_ms']['p95']} ms")


if __name__ == '__main__':
    main()
