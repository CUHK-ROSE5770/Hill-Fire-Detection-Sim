"""Interpolate a converged base YOLO checkpoint with a fine-tuned checkpoint (WiSE-FT)."""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def merge_checkpoints(base_path, finetuned_path, output_path, alpha):
    if not 0.0 <= alpha <= 1.0:
        raise ValueError('alpha must be between 0 and 1')
    base_model = YOLO(base_path)
    ft_model = YOLO(finetuned_path)
    base_sd = {k: v.clone().float() for k, v in base_model.model.state_dict().items()}
    ft_sd = {k: v.clone().float() for k, v in ft_model.model.state_dict().items()}

    ckpt = torch.load(finetuned_path, map_location='cpu', weights_only=False)
    for target_key in ('model', 'ema'):
        module = ckpt.get(target_key)
        if module is None:
            continue
        merged = {}
        for k, v in module.state_dict().items():
            if k in base_sd and k in ft_sd and v.is_floating_point():
                merged[k] = ((1.0 - alpha) * base_sd[k] + alpha * ft_sd[k]).to(v.dtype)
            elif k in base_sd:
                merged[k] = base_sd[k].to(v.dtype)
            else:
                merged[k] = v
        module.load_state_dict(merged)
    if 'train_args' in ckpt and isinstance(ckpt['train_args'], dict):
        ckpt['train_args']['wise_ft_alpha'] = alpha
        ckpt['train_args']['data'] = 'data.yaml'
        ckpt['train_args']['project'] = 'runs'
        ckpt['train_args']['save_dir'] = 'runs/v8n_aerial_aug'
    if hasattr(ckpt['model'], 'args') and isinstance(ckpt['model'].args, dict):
        ckpt['model'].args['wise_ft_alpha'] = alpha
        ckpt['model'].args['data'] = 'data.yaml'
        ckpt['model'].args['project'] = 'runs'
        ckpt['model'].args['save_dir'] = 'runs/v8n_aerial_aug'
    ckpt['model'].criterion = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--finetuned', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--alpha', type=float, default=0.35, help='Weight on the fine-tuned checkpoint')
    args = parser.parse_args()
    merge_checkpoints(args.base, args.finetuned, args.output, args.alpha)
    print(f'Saved WiSE-FT merged checkpoint (alpha={args.alpha}) to {args.output}')


if __name__ == '__main__':
    main()
