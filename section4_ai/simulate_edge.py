"""Replay controlled detector and thermal scenarios against simulated processing budgets."""

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np

from infer import thermal_delta
from perception import ConfirmationGate, Detection
from wind_forecast import project_perimeter


# Fields are ID, whether the incident is real, RGB class, frame count and apparent thermal rise.
SCENARIOS = (
    ('persistent_fire', True, 'fire', 8, 18.),
    ('smoke_with_hot_source', True, 'smoke', 8, 9.),
    ('smoke_without_thermal', True, 'smoke', 8, None),
    ('valley_fog', False, 'smoke', 8, .7),
    ('low_cloud', False, 'smoke', 8, 1.5),
    ('two_frame_glint', False, 'fire', 2, 11.),
    ('brief_heat_spike', False, 'smoke', 8, (0., 0., 13., 0., 0., 0., 0., 0.)),
    ('warm_background_confuser', False, 'fire', 8, 8.),
)


def simulated_detection(label, frame, rise):
    """Create a slightly moving box and an assumed Celsius image aligned with it."""
    offset = frame % 3
    box = (16 + offset, 17, 36 + offset, 35)
    if rise is None:
        return Detection(label, .65, box)
    image = np.full((64, 64), 21., dtype=np.float32)
    x1, y1, x2, y2 = box
    image[y1:y2, x1:x2] += rise
    return Detection(label, .65, box, thermal_delta(image, box))


def simulate_scenario(scenario, fps, service_ms, thermal_hits=2):
    identifier, real, label, hits, rise = scenario
    period = 1000 / fps
    frame_count = 16
    gate = ConfirmationGate(thermal_hits=thermal_hits)
    frame, ready = 0, 0.
    confirmed = []
    candidates = []
    processed = []
    while frame < frame_count:
        start = max(ready, frame * period)
        # One-slot latest-frame buffer. A skipped source frame breaks the gate's consecutive hits.
        frame = min(int((start + 1e-8) // period), frame_count - 1)
        ready = start + service_ms
        detections = []
        if frame < hits:
            current_rise = rise[frame] if isinstance(rise, tuple) else rise
            detections = [simulated_detection(label, frame, current_rise)]
        observations = gate.step(frame, detections)
        if any(observation.consecutive_frames >= 3 for observation in observations):
            candidates.append(ready)
        if any(observation.new_alert for observation in observations):
            confirmed.append(ready)
        processed.append((frame, ready - frame * period))
        frame += 1
    return {'scenario': identifier, 'real_incident': real, 'assumed_thermal_rise_c': rise,
            'visual_candidate': bool(candidates), 'confirmed': bool(confirmed),
            'first_confirmation_ms_from_onset': round(confirmed[0], 2) if confirmed else None,
            'captured': frame_count, 'processed': len(processed), 'finish_ms': max(ready, frame_count * period),
            'capture_to_result_ms': [round(delay, 4) for _, delay in processed]}


def simulate_budget(fps, service_ms, thermal_hits=2):
    trials = [simulate_scenario(scenario, fps, service_ms, thermal_hits) for scenario in SCENARIOS]
    cases = {item['scenario']: item for item in trials}
    latency = [value for item in trials for value in item['capture_to_result_ms']]
    finish = sum(item['finish_ms'] for item in trials)
    processed = sum(item['processed'] for item in trials)
    throughput = 1000 * processed / finish
    p95 = float(np.percentile(latency, 95))
    negative = [trial for trial in trials if not trial['real_incident']]
    positives = [trial for trial in trials if trial['real_incident']]
    return {'camera_fps': fps, 'thermal_hits_required': thermal_hits, 'assumed_service_ms': service_ms,
            'processed_fps': round(throughput, 2),
            'capture_to_result_mean_ms': round(mean(latency), 2), 'capture_to_result_p95_ms': round(p95, 2),
            'single_frame_requirement_in_simulation': bool(throughput > 35 and p95 < 30),
            'dropped_frames': sum(item['captured'] - item['processed'] for item in trials),
            'confirmed_positive_scenarios': sum(item['confirmed'] for item in positives),
            'positive_scenarios': len(positives),
            'negative_visual_candidates': sum(item['visual_candidate'] for item in negative),
            'negative_confirmations': sum(item['confirmed'] for item in negative),
            'negative_scenarios': len(negative),
            'fire_confirmation_ms': cases['persistent_fire']['first_confirmation_ms_from_onset'],
            'scenarios': trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--camera-fps', type=float, default=40.)
    parser.add_argument('--budgets', nargs='+', type=float, default=[18., 24., 29., 33.])
    args = parser.parse_args()
    if not np.isfinite(args.camera_fps) or args.camera_fps <= 0 or (
        any(not np.isfinite(budget) or budget <= 0 for budget in args.budgets)
    ):
        parser.error('Frame rate and service times must be finite and positive')
    result = {'evidence_type': 'deterministic software simulation',
              'input': 'synthetic RGB detector outputs and assumed aligned Celsius arrays',
              'timing': 'assumed per-frame service time with one-slot latest-frame buffer',
              'budgets': [simulate_budget(args.camera_fps, budget) for budget in args.budgets]}
    result['thermal_policy_comparison'] = [simulate_budget(args.camera_fps, min(args.budgets), thermal_hits=hits)
                                           for hits in (1, 2)]
    perimeter = [[-30., -20.], [0., -20.], [30., -20.], [30., 0.], [30., 20.],
                 [0., 20.], [-30., 20.], [-30., 0.]]
    result['advisory_wind_projection'] = {'initial_perimeter_xy_m': perimeter,
                                          **project_perimeter(perimeter, [4., 1.], 15.)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
