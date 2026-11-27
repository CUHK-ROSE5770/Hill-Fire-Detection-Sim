"""Fine-tune the three Section 4 experiment arms on identical data splits."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('YOLO_OFFLINE', 'true')

import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils import torch_utils


EXPERIMENTS = {
    'v8n_base': ('yolov8n.pt', {'mosaic': 0., 'scale': 0., 'flipud': 0., 'translate': 0.}),
    'v8s_base': ('yolov8s.pt', {'mosaic': 0., 'scale': 0., 'flipud': 0., 'translate': 0.}),
    'v8n_aug': ('yolov8n.pt', {'mosaic': 1., 'scale': .35, 'flipud': .10, 'translate': 0.}),
}


def dataset_config(data, output):
    """Resolve data relative to its own YAML, independently of Ultralytics global settings."""
    data = data.resolve()
    spec = yaml.safe_load(data.read_text(encoding='utf-8'))
    spec['path'] = str(data.parent)
    target = output / 'dataset.resolved.yaml'
    target.write_text(yaml.safe_dump(spec, sort_keys=False), encoding='utf-8')
    return target.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--init', required=True, type=Path, help='Directory with official COCO weights')
    parser.add_argument('--only', nargs='+', choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=5770)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--imgsz', type=int, default=256)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--freeze', type=int, default=10)
    args = parser.parse_args()
    args.runs.mkdir(parents=True, exist_ok=True)
    torch_utils.NUM_THREADS = args.threads  # Ultralytics resets this in select_device on CPU.
    torch.set_num_threads(args.threads)
    data = dataset_config(args.data, args.runs)
    for name in args.only:
        path = args.runs.resolve() / name
        if path.exists():
            raise FileExistsError(f'{path} already exists. Use a new --runs directory for a new experiment.')
        weight, aug = EXPERIMENTS[name]
        initial = args.init / weight
        if not initial.is_file():
            raise FileNotFoundError(f'Missing official transfer-learning checkpoint {initial}')
        model = YOLO(initial.resolve())
        model.train(
            data=str(data), project=str(args.runs.resolve()), name=name, device=args.device,
            epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=0,
            freeze=args.freeze, seed=args.seed, deterministic=True, cache=False, amp=False,
            optimizer='AdamW', lr0=.002, lrf=.1, cos_lr=True, warmup_epochs=1., patience=0,
            close_mosaic=0, fliplr=.5, hsv_h=.015, hsv_s=.7, hsv_v=.4,
            plots=False, exist_ok=False, verbose=False, **aug,
        )
        checkpoint = path / 'weights' / 'best.pt'
        if not checkpoint.is_file():
            raise RuntimeError(f'No checkpoint produced for {name}')
        print(json.dumps({'experiment': name, 'checkpoint': str(checkpoint)}), flush=True)


if __name__ == '__main__':
    main()
