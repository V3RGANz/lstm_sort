import numpy as np


def multiclass_accuracy(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    assert isinstance(prediction, np.ndarray)
    assert isinstance(ground_truth, np.ndarray)
    assert prediction.shape == ground_truth.shape
    return np.sum(prediction == ground_truth) / prediction.size
