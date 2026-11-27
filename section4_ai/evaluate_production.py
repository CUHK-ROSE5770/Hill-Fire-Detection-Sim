"""Calibrate 640 px D-Fire models on disjoint calibration images and evaluate once on the fixed 600-image test split."""

import argparse
import csv
import io
import json
import platform
import tarfile
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils import torch_utils

from calibrate_recall import build_calibration, read_manifest, verify_fixed_test
from evaluate import (
    artifact_digest,
    class_frame_metrics,
    confusion_matrix,
    frame_candidate_metrics,
    gate_class_frames,
    predict_split,
    score,
)
from perception import ConfirmationGate, Detection
from prepare_data import CLASSES
from train import dataset_config


PREDICT_FLOOR = {'conf': .002, 'iou': .80, 'max_det': 300}
NMS_IOU = .50
CONF_GRID = (.04, .06, .08, .10, .12, .15, .18, .22, .25)
PEAK_GRID = (.08, .10, .12, .14, .16, .18, .20, .22, .25, .28, .30, .35)


def evaluate_profile(rows, thresholds, peaks, nms_iou=NMS_IOU):
    gated = list(gate_class_frames(rows, thresholds, peaks, nms_iou=nms_iou))
    boxes, cases = score(gated, thresholds)
    confusion = confusion_matrix(boxes['matrix'], cases)
    frames = frame_candidate_metrics(cases)
    cframes = class_frame_metrics(cases)
    return {'boxes': boxes, 'confusion_true_by_predicted': confusion, 'frames': frames, 'class_frames': cframes}, cases


def calibrate_per_class(cal_rows, *, precision_target=.85, recall_target=.90, recall_margin=.01, precision_margin=.015):
    """Select per-class confidence and peak-gate thresholds strictly on the calibration split."""
    selected_conf = []
    selected_peak = []
    # Pre-apply NMS once for fast calibration
    nms_rows = list(gate_class_frames(cal_rows, (.002, .002), (.0, .0), nms_iou=NMS_IOU))
    for cls_id, label in enumerate(CLASSES):
        best = None
        for conf in CONF_GRID:
            for peak in PEAK_GRID:
                if peak < conf:
                    continue
                thresholds = (conf, .15) if cls_id == 0 else (.10, conf)
                peaks = (peak, .20) if cls_id == 0 else (.15, peak)
                report, _ = evaluate_profile(nms_rows, thresholds, peaks, nms_iou=None)
                cf = report['class_frames'][label]
                box_p = report['boxes']['precision'][label]
                p, r = cf['localized_precision'], cf['localized_recall']
                meets_margin = (r >= recall_target + recall_margin) and (p >= precision_target + precision_margin)
                meets_exact = (r >= recall_target) and (p >= precision_target)
                key = (int(meets_margin), int(meets_exact), int(r >= recall_target), int(p >= precision_target),
                       p + r + 0.25 * box_p, -abs(r - (recall_target + .02)))
                if best is None or key > best[0]:
                    best = (key, conf, peak)
        selected_conf.append(best[1])
        selected_peak.append(best[2])
    return tuple(selected_conf), tuple(selected_peak)


def evaluate_aof_temporal(model, archive, imgsz, thresholds, peaks, device, nms_iou=NMS_IOU):
    """Replay real continuous AoF test clips through the 3-frame RGB ConfirmationGate."""
    with tarfile.open(archive) as tf:
        members = {
            m.name.split('/')[-1]: m
            for m in tf.getmembers()
            if m.isfile() and not m.name.split('/')[-1].startswith('._')
        }
        # Scan AoF6723..8383 and segment visually continuous clips (MAD < 10.0) of >= 5 frames
        frames = []
        prev = None
        for idx in range(6723, 8384):
            fn, lf = f'AoF{idx:05d}.jpg', f'AoF{idx:05d}.txt'
            if fn not in members or lf not in members:
                prev = None
                continue
            raw_img = tf.extractfile(members[fn]).read()
            raw_lbl = tf.extractfile(members[lf]).read().decode('utf-8-sig').strip().splitlines()
            has_gt = any(ln.strip() for ln in raw_lbl)
            arr = np.frombuffer(raw_img, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            small = cv2.resize(img, (48, 27)).astype(np.float32)
            mad = float(np.abs(small - prev).mean()) if prev is not None else 999.0
            prev = small
            frames.append((idx, fn, has_gt, mad, img))

    clips, cur = [], []
    for item in frames:
        if cur and item[3] < 10.0 and cur[-1][2] == item[2] and item[0] == cur[-1][0] + 1:
            cur.append(item)
        else:
            if len(cur) >= 5:
                clips.append(cur)
            cur = [item]
    if len(cur) >= 5:
        clips.append(cur)

    pos_clips = tp_clips = neg_clips = fp_clips = 0
    clip_details = []
    floor = min(thresholds)
    for clip in clips:
        # Cap each clip to first 12 frames for rapid evaluation of onset confirmation
        sub = clip[:12]
        is_pos = sub[0][2]
        gate = ConfirmationGate(confidence=floor, frames=3, association_iou=.30)
        triggered = False
        first_hit = None
        for f_id, (_, fn, _, _, img) in enumerate(sub):
            res = model.predict(img, imgsz=imgsz, conf=floor, iou=nms_iou, max_det=100,
                                device=device, verbose=False, rect=False)[0]
            dets = []
            for x1, y1, x2, y2, sc, cls in res.boxes.data.cpu().tolist():
                cls_i = int(cls)
                if x2 > x1 and y2 > y1 and sc >= thresholds[cls_i]:
                    dets.append((cls_i, Detection(CLASSES[cls_i], sc, (x1, y1, x2, y2))))
            kept = []
            for cls_i in (0, 1):
                c_dets = [d for ci, d in dets if ci == cls_i]
                if c_dets and max(d.confidence for d in c_dets) >= peaks[cls_i]:
                    kept.extend(c_dets)
            obs = gate.step(f_id, kept)
            if any(o.consecutive_frames >= 3 for o in obs):
                triggered = True
                if first_hit is None:
                    first_hit = f_id + 1
        if is_pos:
            pos_clips += 1
            tp_clips += int(triggered)
        else:
            neg_clips += 1
            fp_clips += int(triggered)
        clip_details.append({
            'start': sub[0][1], 'end': sub[-1][1], 'frames': len(sub),
            'positive': is_pos, 'rgb_3frame_candidate': triggered, 'first_candidate_frame': first_hit,
        })
    precision = tp_clips / max(1, tp_clips + fp_clips)
    recall = tp_clips / max(1, pos_clips)
    return {
        'clips_total': len(clips),
        'positive_clips': pos_clips,
        'negative_clips': neg_clips,
        'tp_clips': tp_clips,
        'fp_clips': fp_clips,
        'clip_precision': precision,
        'clip_recall': recall,
        'negative_clip_alarm_rate': fp_clips / max(1, neg_clips),
        'clips': clip_details,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path, help='Official D-Fire test.tar')
    parser.add_argument('--data', required=True, type=Path, help='Fixed 4,500-image benchmark root')
    parser.add_argument('--model', action='append', required=True,
                        help='Named checkpoint in format name=path[:imgsz[:tta]]')
    parser.add_argument('--output', required=True, type=Path, help='Output directory for benchmark JSON/CSV')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--temporal-model', help='Model name on which to run real AoF 3-frame temporal verification')
    parser.add_argument('--save-boxes', nargs='*', default=[], help='Model names whose test boxes JSON should be saved')
    args = parser.parse_args()

    manifests = Path(__file__).with_name('data')
    fixed_test = [row for row in read_manifest(manifests / 'split_manifest.csv') if row['split'] == 'test']
    calibration = read_manifest(manifests / 'recall_calibration_manifest.csv')
    verify_fixed_test(args.data, fixed_test)

    args.output.mkdir(parents=True, exist_ok=True)
    cal_dir = args.output / 'calibration'
    if not (cal_dir / 'images').exists():
        build_calibration(args.archive, cal_dir, calibration, fixed_test)

    torch_utils.NUM_THREADS = args.threads
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    resolved_yaml = dataset_config(args.data / 'data.yaml', args.output)

    models_out = {}
    temporal_out = None
    for spec in args.model:
        name, rest = spec.split('=', 1)
        parts = rest.split(':')
        weights = Path(parts[0])
        imgsz = int(parts[1]) if len(parts) > 1 and parts[1] else 640
        tta = parts[2].lower() == 'tta' if len(parts) > 2 else False

        print(f'=== Evaluating {name} ({weights.name}, imgsz={imgsz}, tta={tta}) ===', flush=True)
        model = YOLO(weights)
        if model.names != dict(enumerate(CLASSES)):
            raise ValueError(f'Unexpected class IDs for {weights}: {model.names}')
        params = sum(p.numel() for p in model.model.parameters())

        cal_rows = predict_split(model, args.output.resolve(), 'calibration', imgsz, args.device,
                                 augment=tta, **PREDICT_FLOOR)
        thresholds, peaks = calibrate_per_class(cal_rows)
        cal_report, _ = evaluate_profile(cal_rows, thresholds, peaks, nms_iou=NMS_IOU)

        test_rows = predict_split(model, args.data.resolve(), 'test', imgsz, args.device,
                                  augment=tta, **PREDICT_FLOOR)
        test_report, test_cases = evaluate_profile(test_rows, thresholds, peaks, nms_iou=NMS_IOU)
        shared_report, _ = evaluate_profile(test_rows, (.10, .10), (.0, .0), nms_iou=NMS_IOU)

        latency = np.asarray([sum(item[-1].values()) for item in test_rows[10:]])
        val_metrics = model.val(
            data=str(resolved_yaml), split='test', imgsz=imgsz, batch=1, device=args.device, workers=0,
            plots=False, conf=.001, iou=.70, max_det=100, rect=False,
            project=str(args.output.resolve()), name=f'{name}_ap', verbose=False,
        )
        models_out[name] = {
            'weights': weights.name,
            'checkpoint': artifact_digest(weights),
            'parameters': params,
            'imgsz': imgsz,
            'tta': tta,
            'nms_iou': NMS_IOU,
            'thresholds': dict(zip(CLASSES, thresholds)),
            'peak_thresholds': dict(zip(CLASSES, peaks)),
            'standard_ap': {
                'map50': float(val_metrics.box.map50),
                'map50_95': float(val_metrics.box.map),
                'class_ap50': {CLASSES[int(c)]: float(ap) for c, ap in zip(val_metrics.box.ap_class_index, val_metrics.box.ap50)},
            },
            'calibration': cal_report,
            'test': test_report,
            'test_shared_conf10': shared_report,
            'pipeline_mean_ms': float(latency.mean()),
            'pipeline_p95_ms': float(np.percentile(latency, 95)),
            'pipeline_fps': float(1000.0 / latency.mean()),
        }
        if name in args.save_boxes:
            (args.output / f'{name}_test_boxes.json').write_text(
                json.dumps(test_cases, separators=(',', ':')) + '\n', encoding='utf-8'
            )
        tf_m = test_report['frames']
        cf_m = test_report['class_frames']
        bx_m = test_report['boxes']
        print(
            f"  mAP50={val_metrics.box.map50*100:.1f}% | "
            f"Overall Frame P/R={tf_m['localized_candidate_precision']*100:.1f}%/{tf_m['localized_frame_recall']*100:.1f}% | "
            f"Smoke Frame P/R={cf_m['smoke']['localized_precision']*100:.1f}%/{cf_m['smoke']['localized_recall']*100:.1f}% | "
            f"Fire Frame P/R={cf_m['fire']['localized_precision']*100:.1f}%/{cf_m['fire']['localized_recall']*100:.1f}% | "
            f"Neg={bx_m['negative_image_alert_rate']*100:.1f}% | FPS={1000.0/latency.mean():.1f}",
            flush=True,
        )
        if args.temporal_model == name:
            print(f'=== Running real AoF 3-frame temporal verification for {name} ===', flush=True)
            temporal_out = evaluate_aof_temporal(model, args.archive, imgsz, thresholds, peaks, args.device, NMS_IOU)
            print(
                f"  AoF clips ({temporal_out['positive_clips']} pos / {temporal_out['negative_clips']} bg): "
                f"P={temporal_out['clip_precision']*100:.1f}%, R={temporal_out['clip_recall']*100:.1f}%, "
                f"NegAlarm={temporal_out['negative_clip_alarm_rate']*100:.1f}%",
                flush=True,
            )

    bundle = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'device': args.device,
        'torch': torch.__version__,
        'ultralytics': ultralytics.__version__,
        'threads': args.threads,
        'calibration_images': len(calibration),
        'test_images': len(fixed_test),
        'models': models_out,
        'aof_temporal_verification': temporal_out,
    }
    (args.output / 'production_benchmark.json').write_text(json.dumps(bundle, indent=2) + '\n', encoding='utf-8')

    with (args.output / 'production_benchmark.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow((
            'model', 'parameters', 'imgsz', 'map50', 'map50_95',
            'frame_precision', 'frame_recall',
            'smoke_frame_precision', 'smoke_frame_recall',
            'fire_frame_precision', 'fire_frame_recall',
            'smoke_box_precision', 'smoke_box_recall',
            'fire_box_precision', 'fire_box_recall',
            'small_smoke_recall', 'negative_frame_rate', 'cpu_fps',
        ))
        for name, row in models_out.items():
            t = row['test']
            writer.writerow((
                name, row['parameters'], row['imgsz'],
                row['standard_ap']['map50'], row['standard_ap']['map50_95'],
                t['frames']['localized_candidate_precision'], t['frames']['localized_frame_recall'],
                t['class_frames']['smoke']['localized_precision'], t['class_frames']['smoke']['localized_recall'],
                t['class_frames']['fire']['localized_precision'], t['class_frames']['fire']['localized_recall'],
                t['boxes']['precision']['smoke'], t['boxes']['recall']['smoke'],
                t['boxes']['precision']['fire'], t['boxes']['recall']['fire'],
                t['boxes']['small_recall']['smoke'], t['boxes']['negative_image_alert_rate'],
                row['pipeline_fps'],
            ))


if __name__ == '__main__':
    main()
