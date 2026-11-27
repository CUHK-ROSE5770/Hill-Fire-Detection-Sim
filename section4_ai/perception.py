"""Per-UAV RGB track confirmation and thermal evidence gate for simulator observations."""

from dataclasses import dataclass
from math import isfinite


def iou(a, b):
    dx = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    dy = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    area = dx * dy
    return area / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area)


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    thermal_delta_c: float | None = None

    def __post_init__(self):
        x1, y1, x2, y2 = self.xyxy
        if self.label not in {'smoke', 'fire'} or not 0. <= self.confidence <= 1.:
            raise ValueError('Expected smoke or fire with a confidence between 0 and 1')
        if not all(map(isfinite, self.xyxy)) or not (0 <= x1 < x2 and 0 <= y1 < y2):
            raise ValueError('Expected a positive-area box in RGB pixel coordinates')
        if self.thermal_delta_c is not None and not isfinite(self.thermal_delta_c):
            raise ValueError('Use None for unavailable thermal evidence')


@dataclass(frozen=True)
class Observation:
    track_id: int
    frame_id: int
    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    bearing_pixel: tuple[float, float]
    consecutive_frames: int
    thermal_delta_c: float | None
    state: str
    new_alert: bool


class ConfirmationGate:
    """One instance per camera. Only consecutive, spatially associated RGB hits count."""

    def __init__(self, confidence=.10, frames=3, association_iou=.30, min_thermal_delta_c=5.):
        if not 0 <= confidence <= 1 or frames < 1 or not 0 < association_iou <= 1:
            raise ValueError('Invalid RGB or tracking thresholds')
        if not isfinite(min_thermal_delta_c) or min_thermal_delta_c <= 0:
            raise ValueError('Thermal temperature threshold must be positive')
        self.confidence = confidence
        self.frames = frames
        self.association_iou = association_iou
        self.min_thermal_delta_c = min_thermal_delta_c
        self._last_frame = None
        self._next_id = 1
        self._tracks = []

    def step(self, frame_id, detections):
        """Return observations in the current frame. Downstream verification can use RGB-only candidates."""
        if self._last_frame is not None and frame_id <= self._last_frame:
            raise ValueError('Frame IDs must be strictly increasing')
        previous = self._tracks if self._last_frame is not None and frame_id == self._last_frame + 1 else []
        available = {track.track_id: (track, alerted) for track, alerted in previous}
        tracks = []
        for detection in sorted(detections, key=lambda item: -item.confidence):
            if detection.confidence < self.confidence:
                continue
            match_iou, track_id = max(
                ((iou(track.xyxy, detection.xyxy), track.track_id)
                 for track, _ in available.values() if track.label == detection.label), default=(0., -1)
            )
            if match_iou >= self.association_iou:
                track, alerted = available.pop(track_id)
                count = track.consecutive_frames + 1
            else:
                track_id, count, alerted = self._next_id, 1, False
                self._next_id += 1
            delta = detection.thermal_delta_c
            confirmed = count >= self.frames and delta is not None and delta >= self.min_thermal_delta_c
            state = 'confirmed' if confirmed else ('rgb_candidate' if count >= self.frames else 'tracking')
            x1, y1, x2, y2 = detection.xyxy
            observation = Observation(track_id, frame_id, detection.label, detection.confidence, detection.xyxy,
                                      ((x1 + x2) / 2, (y1 + y2) / 2), count, delta, state,
                                      confirmed and not alerted)
            tracks.append((observation, alerted or confirmed))
        self._last_frame, self._tracks = frame_id, tracks
        return [observation for observation, _ in tracks]
