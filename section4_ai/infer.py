"""Replay an RGB image or video into the Section 4 per-UAV observation contract."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from perception import ConfirmationGate, Detection


def thermal_delta(image, box):
    """Estimate the warmest part of the RGB ROI against a local thermal background ring."""
    h, w = image.shape
    x1, y1, x2, y2 = box
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    pad = max(4, max(x2 - x1, y2 - y1) // 2)
    left, top = max(0, x1 - pad), max(0, y1 - pad)
    ring = image[top:min(h, y2 + pad), left:min(w, x2 + pad)].copy()
    core = image[y1:y2, x1:x2]
    ring[y1 - top:y2 - top, x1 - left:x2 - left] = np.nan
    foreground, background = core[np.isfinite(core)], ring[np.isfinite(ring)]
    if foreground.size < 4 or background.size < 4:
        return None
    return float(np.percentile(foreground, 90) - np.median(background))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True, help='One RGB image or video')
    parser.add_argument('--output', type=Path, required=True, help='New JSONL observation file')
    parser.add_argument('--thermal', type=Path, help='Aligned and time-synchronized floating-point Celsius frames (.npy)')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--imgsz', type=int, default=256)
    parser.add_argument('--conf', type=float, default=.10)
    parser.add_argument('--smoke-conf', type=float)
    parser.add_argument('--fire-conf', type=float)
    parser.add_argument('--nms-iou', type=float, default=.70)
    parser.add_argument('--max-det', type=int, default=100)
    parser.add_argument('--frame-conf', type=float, default=0., help='Minimum peak score among accepted RGB boxes')
    parser.add_argument('--thermal-delta-c', type=float, default=5.)
    parser.add_argument('--thermal-hits', type=int, default=1, help='Warm observations required in the latest three RGB hits')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    thermal = np.load(args.thermal, mmap_mode='r', allow_pickle=False) if args.thermal else None
    if thermal is not None and (thermal.ndim != 3 or thermal.dtype.kind != 'f'):
        raise ValueError('Thermal input must be a float array of shape (frames, height, width)')
    limits = {label: args.conf if threshold is None else threshold
              for label, threshold in (('smoke', args.smoke_conf), ('fire', args.fire_conf))}
    if any(not 0 <= threshold <= 1 for threshold in [*limits.values(), args.frame_conf]):
        parser.error('Confidence thresholds must be between 0 and 1')
    threshold_floor = min(limits.values())
    gate = ConfirmationGate(confidence=threshold_floor, min_thermal_delta_c=args.thermal_delta_c,
                            thermal_hits=args.thermal_hits)
    model = YOLO(args.weights, task='detect')
    stream = model.predict(source=str(args.source), stream=True, imgsz=args.imgsz, conf=threshold_floor,
                           device=args.device, verbose=False, rect=False, iou=args.nms_iou, max_det=args.max_det)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    with temporary.open('x', encoding='utf-8') as file:
        try:
            count = 0
            for frame_id, result in enumerate(stream):
                if thermal is not None and (frame_id >= len(thermal) or thermal.shape[1:] != result.orig_shape):
                    raise ValueError(f'Thermal alignment or frame count mismatch at RGB frame {frame_id}')
                detections = []
                for x1, y1, x2, y2, score, cls in result.boxes.data.cpu().tolist():
                    label = result.names[int(cls)]
                    if x2 <= x1 or y2 <= y1 or score < limits[label]:
                        continue
                    box = x1, y1, x2, y2
                    delta = None if thermal is None else thermal_delta(thermal[frame_id], box)
                    detections.append(Detection(label, score, box, delta))
                if max((item.confidence for item in detections), default=0.) < args.frame_conf:
                    detections.clear()
                observations = gate.step(frame_id, detections)
                file.write(json.dumps({'frame_id': frame_id, 'height': result.orig_shape[0],
                                       'width': result.orig_shape[1],
                                       'observations': [asdict(item) for item in observations]}) + '\n')
                count += 1
            if thermal is not None and count != len(thermal):
                raise ValueError('Thermal input contains more frames than the RGB source')
        except Exception:
            file.close()
            temporary.unlink()
            raise
    temporary.replace(args.output)


if __name__ == '__main__':
    main()
