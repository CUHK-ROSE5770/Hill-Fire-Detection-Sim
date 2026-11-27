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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--multiscale', type=Path)
    parser.add_argument('--frame-gate', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--precision', type=float, default=.90)
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
    result = {'scope': 'offline RGB box detection at same-class IoU 0.50',
              'project_review_targets': {'precision': args.precision, 'recall': args.recall},
              'profiles': profiles, 'physical_jetson_acceptance': 'unmeasured',
              'field_event_acceptance': 'unmeasured'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(profiles, indent=2))


if __name__ == '__main__':
    main()
