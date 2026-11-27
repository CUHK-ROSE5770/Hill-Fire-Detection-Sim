"""Check joint precision and recall targets using measured, held-out box counts."""

import argparse
import json
from pathlib import Path

from prepare_data import CLASSES


def check_targets(boxes, precision_target, recall_target):
    matrix = boxes['matrix']
    checks = {}
    for index, label in enumerate(CLASSES):
        matched = matrix[index][index]
        predictions = sum(row[index] for row in matrix)
        annotations = sum(matrix[index])
        if not annotations:
            raise ValueError(f'The holdout needs annotated {label} boxes')
        precision = matched / max(1, predictions)
        recall = matched / annotations
        checks[label] = {'matched': matched, 'predicted': predictions, 'annotated': annotations,
                         'precision': precision, 'recall': recall,
                         'hits_precision_target': precision > precision_target,
                         'hits_recall_target': recall > recall_target}
    return {'classes': checks,
            'hits_both_targets_for_both_classes': all(check['hits_precision_target'] and check['hits_recall_target']
                                                      for check in checks.values())}


def check_operational_targets(frames, class_frames, precision_target, recall_target):
    """Verify operational localized-frame precision/recall targets overall and per class."""
    overall_p = frames['localized_candidate_precision']
    overall_r = frames['localized_frame_recall']
    overall = {
        'precision': overall_p,
        'recall': overall_r,
        'hits_precision_target': overall_p >= precision_target,
        'hits_recall_target': overall_r >= recall_target,
    }
    classes = {}
    for label in CLASSES:
        entry = class_frames[label]
        p, r = entry['localized_precision'], entry['localized_recall']
        classes[label] = {
            'localized_frames': entry['localized_frames'],
            'predicted_frames': entry['predicted_frames'],
            'positive_frames': entry['positive_frames'],
            'precision': p,
            'recall': r,
            'presence_precision': entry['presence_precision'],
            'presence_recall': entry['presence_recall'],
            'hits_precision_target': p >= precision_target,
            'hits_recall_target': r >= recall_target,
        }
    hits_all = (
        overall['hits_precision_target']
        and overall['hits_recall_target']
        and all(item['hits_precision_target'] and item['hits_recall_target'] for item in classes.values())
    )
    return {'overall': overall, 'classes': classes, 'hits_all_operational_targets': hits_all}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--multiscale', type=Path)
    parser.add_argument('--frame-gate', type=Path)
    parser.add_argument('--production', type=Path, help='Production 640 px benchmark JSON with per-class frame metrics')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--precision', type=float, default=.85)
    parser.add_argument('--recall', type=float, default=.90)
    args = parser.parse_args()
    if not (0 < args.precision <= 1 and 0 < args.recall <= 1):
        parser.error('Precision and recall targets must be between zero and one')
    reference = json.loads(args.reference.read_text(encoding='utf-8'))
    profiles = {'single_scale_recall': check_targets(reference['test_operating_point'], args.precision, args.recall)}
    if args.multiscale:
        multiscale = json.loads(args.multiscale.read_text(encoding='utf-8'))
        profiles['dual_scale'] = check_targets(multiscale['test']['boxes'], args.precision, args.recall)
    if args.frame_gate:
        gated = json.loads(args.frame_gate.read_text(encoding='utf-8'))
        profiles['frame_gated_boxes'] = check_targets(gated['test']['boxes'], args.precision, args.recall)
    operational = {}
    if args.production:
        prod = json.loads(args.production.read_text(encoding='utf-8'))
        for name, entry in prod['models'].items():
            test = entry['test']
            operational[name] = check_operational_targets(
                test['frames'], test['class_frames'], args.precision, args.recall
            )
            profiles[f'{name}_boxes'] = check_targets(test['boxes'], args.precision, args.recall)
    result = {
        'scope': 'offline RGB localized-frame and single-box detection at same-class IoU 0.50',
        'industrial_production_targets': {'precision': args.precision, 'recall': args.recall},
        'operational_localized_frame_profiles': operational,
        'box_level_profiles': profiles,
        'physical_jetson_acceptance': 'unmeasured',
        'field_event_acceptance': 'unmeasured',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'operational': operational, 'box_level': profiles}, indent=2))


if __name__ == '__main__':
    main()
