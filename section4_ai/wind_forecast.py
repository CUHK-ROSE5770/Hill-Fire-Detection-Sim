"""Make an advisory short-horizon perimeter projection in a local metric frame."""

import numpy as np


def project_perimeter(points, wind_xy_mps, seconds, *, base_spread=.15, wind_factor=.08, max_spread=.80):
    """Expand the front along center-to-vertex directions and rank downwind look points."""
    points = np.asarray(points, dtype=float)
    wind = np.asarray(wind_xy_mps, dtype=float)
    settings = np.asarray((seconds, base_spread, wind_factor, max_spread))
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or wind.shape != (2,)
            or not np.isfinite(points).all() or not np.isfinite(wind).all()
            or not np.isfinite(settings).all() or np.any(settings < 0) or max_spread < base_spread):
        raise ValueError('Expected finite XY points, a wind vector and nonnegative spread settings')
    center = points.mean(axis=0)
    radial = points - center
    directions = radial / np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
    downwind = directions @ wind
    speed = np.minimum(max_spread, base_spread + wind_factor * np.maximum(downwind, 0))
    forecast = points + seconds * speed[:, None] * directions
    wind_speed = np.linalg.norm(wind)
    priorities = np.flatnonzero(downwind >= .5 * wind_speed) if wind_speed > 0 else np.array([], dtype=int)
    return {'prediction': forecast.tolist(), 'priority_indices': priorities.tolist(),
            'assumptions': {'seconds': seconds, 'base_spread_mps': base_spread, 'wind_factor': wind_factor,
                            'max_spread_mps': max_spread, 'wind_xy_mps': wind.tolist()}}
