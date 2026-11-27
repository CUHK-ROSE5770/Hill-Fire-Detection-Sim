import unittest

from evaluate import confusion_matrix, score


class ConfusionMatrixTests(unittest.TestCase):
    def test_wrong_class_cannot_take_a_correct_match(self):
        truth = [(0, 0, 0, 10, 10, True), (1, 12, 0, 22, 10, True)]
        predictions = [[0, 0, 10, 10, .9, 1], [12, 0, 22, 10, .8, 0], [0, 0, 10, 10, .7, 0]]
        result, cases = score([('frame.jpg', 24, 12, truth, predictions, {})], .1)
        self.assertEqual(result['recall'], {'smoke': 1.0, 'fire': 0.0})
        self.assertEqual(confusion_matrix(result['matrix'], cases), [[1, 0, 0], [1, 0, 0], [0, 1, 0]])
