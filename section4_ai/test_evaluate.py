import unittest

from evaluate import confusion_matrix, score


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
