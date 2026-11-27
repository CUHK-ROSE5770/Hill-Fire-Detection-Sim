"""Fine-tune 640 px D-Fire YOLO detectors with aerial small-target augmentation and hard negatives."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('YOLO_OFFLINE', 'true')

import torch
from ultralytics import YOLO
from ultralytics.utils import torch_utils


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path, help='YOLO dataset YAML (strictly from official train split)')
    parser.add_argument('--weights', required=True, type=Path, help='Base D-Fire 640 px checkpoint (.pt)')
    parser.add_argument('--runs', required=True, type=Path, help='Output runs directory')
    parser.add_argument('--name', required=True, help='Run name')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=5770)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--lr0', type=float, default=1.2e-4)
    parser.add_argument('--close-mosaic', type=int, default=2)
    parser.add_argument('--mosaic', type=float, default=1.0)
    parser.add_argument('--scale', type=float, default=0.50)
    parser.add_argument('--flipud', type=float, default=0.05)
    parser.add_argument('--degrees', type=float, default=8.0)
    parser.add_argument('--copy-paste', type=float, default=0.10)
    parser.add_argument('--cls-gain', type=float, default=0.70)
    args = parser.parse_args()

    args.runs.mkdir(parents=True, exist_ok=True)
    run_dir = args.runs.resolve() / args.name
    if run_dir.exists():
        raise FileExistsError(f'{run_dir} already exists. Use a fresh --name or --runs directory.')

    torch_utils.NUM_THREADS = args.threads
    torch.set_num_threads(args.threads)

    model = YOLO(args.weights.resolve())
    model.train(
        data=str(args.data.resolve()),
        project=str(args.runs.resolve()),
        name=args.name,
        device=args.device,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=0,
        freeze=0,
        seed=args.seed,
        deterministic=True,
        cache=False,
        amp=False,
        optimizer='AdamW',
        lr0=args.lr0,
        lrf=0.05,
        cos_lr=True,
        warmup_epochs=0.5,
        weight_decay=5e-4,
        patience=0,
        cls=args.cls_gain,
        box=7.5,
        dfl=1.5,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        scale=args.scale,
        flipud=args.flipud,
        fliplr=0.5,
        degrees=args.degrees,
        translate=0.1,
        copy_paste=args.copy_paste,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
        plots=False,
        exist_ok=False,
        verbose=True,
    )
    best = run_dir / 'weights' / 'best.pt'
    last = run_dir / 'weights' / 'last.pt'
    print(json.dumps({'run': args.name, 'best': str(best), 'last': str(last)}), flush=True)


if __name__ == '__main__':
    main()
