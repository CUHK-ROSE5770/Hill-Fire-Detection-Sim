import unittest

from simulate_edge import SCENARIOS, simulate_budget, simulate_scenario
from wind_forecast import project_perimeter


class SimulationTests(unittest.TestCase):
    def test_fire_confirmation_counts_acquisition_time(self):
        result = simulate_scenario(SCENARIOS[0], fps=40., service_ms=24.)
        self.assertTrue(result['confirmed'])
        self.assertEqual(result['first_confirmation_ms_from_onset'], 74.)

    def test_controlled_misses_and_false_candidates_are_reported(self):
        result = simulate_budget(40., 24.)
        cases = {row['scenario']: row for row in result['scenarios']}
        self.assertTrue(cases['smoke_without_thermal']['visual_candidate'])
        self.assertFalse(cases['smoke_without_thermal']['confirmed'])
        self.assertFalse(cases['valley_fog']['confirmed'])
        self.assertFalse(cases['two_frame_glint']['visual_candidate'])
        self.assertTrue(cases['warm_background_confuser']['confirmed'])

    def test_over_budget_increases_latency_and_drops_frames(self):
        result = simulate_budget(40., 33.)
        self.assertGreater(result['dropped_frames'], 0)
        self.assertGreater(result['capture_to_result_p95_ms'], 30.)
        self.assertFalse(result['single_frame_requirement_in_simulation'])

    def test_two_warm_hits_reject_a_one_frame_thermal_spike(self):
        reference = simulate_budget(40., 24., thermal_hits=1)
        robust = simulate_budget(40., 24., thermal_hits=2)
        self.assertEqual(reference['confirmed_positive_scenarios'], robust['confirmed_positive_scenarios'])
        self.assertEqual(reference['negative_confirmations'] - robust['negative_confirmations'], 1)

    def test_wind_prioritizes_downwind_points(self):
        points = [[10., 0.], [0., 10.], [-10., 0.], [0., -10.]]
        forecast = project_perimeter(points, [4., 0.], 10.)
        self.assertEqual(forecast['priority_indices'], [0])
        self.assertGreater(forecast['prediction'][0][0] - 10., -10. - forecast['prediction'][2][0])
