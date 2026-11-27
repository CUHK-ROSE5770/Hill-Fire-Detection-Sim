import unittest

from assess_quality import check_targets
from evaluate import confusion_matrix, frame_candidate_metrics, gate_frames, score
from evaluate_multiscale import merge_multiscale


class ConfusionMatrixTests(unittest.TestCase):
    def test_wrong_class_cannot_take_a_correct_match(self):
        truth = [(0, 0, 0, 10, 10, True), (1, 12, 0, 22, 10, True)]
        predictions = [[0, 0, 10, 10, .9, 1], [12, 0, 22, 10, .8, 0], [0, 0, 10, 10, .7, 0]]
        result, cases = score([('frame.jpg', 24, 12, truth, predictions, {})], .1)
        self.assertEqual(result['recall'], {'smoke': 1.0, 'fire': 0.0})
        self.assertEqual(confusion_matrix(result['matrix'], cases), [[1, 0, 0], [1, 0, 0], [0, 1, 0]])

    def test_class_specific_confidences_use_the_detection_class(self):
        truth = [(0, 0, 0, 10, 10, False), (1, 12, 0, 22, 10, False)]
        predictions = [[0, 0, 10, 10, .02, 0], [12, 0, 22, 10, .01, 1]]
        result, _ = score([('frame.jpg', 24, 12, truth, predictions, {})], (.03, .002))
        self.assertEqual(result['recall'], {'smoke': 0.0, 'fire': 1.0})

    def test_frame_precision_requires_a_localized_match(self):
        truth = [(0, 0, 0, 10, 10, False)]
        rows = [('matched.jpg', 60, 30, truth, [[0, 0, 10, 10, .8, 0]], {}),
                ('unmatched.jpg', 60, 30, truth, [[30, 3, 40, 13, .7, 0]], {}),
                ('empty.jpg', 60, 30, [], [[30, 3, 40, 13, .6, 0]], {})]
        _, cases = score(rows, .10)
        metrics = frame_candidate_metrics(cases)
        self.assertEqual(metrics['localized_frame_recall'], .5)
        self.assertEqual(metrics['localized_candidate_precision'], 1 / 3)

    def test_second_scale_adds_only_nonduplicate_fire(self):
        main = [[0, 0, 10, 10, .8, 1], [20, 0, 30, 10, .7, 0]]
        extra = [[0, 0, 10, 10, .6, 1], [40, 0, 50, 10, .5, 1], [60, 0, 70, 10, .4, 0]]
        self.assertEqual(merge_multiscale(main, extra), [*main, extra[1]])

    def test_high_recall_cannot_pass_a_failed_precision_target(self):
        result = check_targets({'matrix': [[94, 0, 6], [0, 92, 8], [8, 900, 0]]}, .90, .90)
        self.assertTrue(result['classes']['fire']['hits_recall_target'])
        self.assertFalse(result['classes']['fire']['hits_precision_target'])
        self.assertFalse(result['hits_both_targets_for_both_classes'])

    def test_peak_gate_only_counts_boxes_over_their_class_threshold(self):
        rows = [('quiet.jpg', 60, 30, [], [[0, 0, 10, 10, .08, 1], [20, 0, 30, 10, .19, 0]], {})]
        gated = list(gate_frames(rows, (.20, .002), .125))
        self.assertEqual(gated[0][4], [])
