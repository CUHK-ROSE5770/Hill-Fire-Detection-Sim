import unittest

import numpy as np

from infer import thermal_delta
from perception import ConfirmationGate, Detection


class ConfirmationTests(unittest.TestCase):
    def test_requires_three_consecutive_hits_and_thermal_evidence(self):
        gate = ConfirmationGate()
        rgb = Detection('smoke', .6, (10, 20, 50, 90))
        warm = Detection('smoke', .6, (12, 21, 51, 89), 8.)
        self.assertEqual(gate.step(1, [warm])[0].state, 'tracking')
        self.assertEqual(gate.step(2, [warm])[0].state, 'tracking')
        observation = gate.step(3, [rgb])[0]
        self.assertEqual(observation.state, 'rgb_candidate')
        self.assertFalse(observation.new_alert)
        self.assertTrue(gate.step(4, [warm])[0].new_alert)
        self.assertFalse(gate.step(5, [warm])[0].new_alert)
        self.assertEqual(gate.step(7, [warm])[0].consecutive_frames, 1)

    def test_independent_targets_and_missed_frames(self):
        gate = ConfirmationGate()
        near = Detection('fire', .6, (1, 1, 10, 10), 8.)
        far = Detection('fire', .6, (21, 21, 30, 30), 8.)
        weak = Detection('smoke', .01, (1, 1, 10, 10), 8.)
        a, b = gate.step(0, [near, far, weak])
        self.assertNotEqual(a.track_id, b.track_id)
        gate.step(1, [near, far])
        self.assertTrue(gate.step(2, [far])[0].new_alert)
        self.assertEqual(gate.step(3, [near])[0].state, 'tracking')
        with self.assertRaises(ValueError):
            gate.step(3, [near])

    def test_aligned_temperature_uses_surrounding_background(self):
        image = np.full((32, 32), 20., dtype=np.float32)
        image[8:16, 8:16] = 30.
        self.assertEqual(thermal_delta(image, (8, 8, 16, 16)), 10.)
        self.assertIsNone(thermal_delta(image, (0, 0, 32, 32)))
        image[:] = np.nan
        self.assertIsNone(thermal_delta(image, (8, 8, 16, 16)))


if __name__ == '__main__':
    unittest.main()
