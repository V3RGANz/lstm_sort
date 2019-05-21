import numpy as np


class Loss:
    def compute(self, predictions: np.ndarray, target: np.ndarray) -> (np.ndarray, np.ndarray):
        raise Exception('Not implemented')


class CrossEntropy(Loss):
    def compute(self, predictions: np.ndarray, target_index: np.ndarray) -> (np.ndarray, np.ndarray):
        """compute cross-entropy loss
        
        Arguments:
        ----------
            predictions {np.ndarray} -- values predicted by model
            shape is either (N,) or (batch_size, N)
            target_index {np.ndarray} -- ground truth values,
            shape is either (1,) or (batch_size,) respectevely
        
        Returns:
        --------
            (np.ndarray, np.ndarray) -- loss value and loss derivative of predictions
        """
        assert isinstance(predictions, np.ndarray)
        assert isinstance(target_index, np.ndarray)

        reshaped_predictions = predictions.reshape((-1, predictions.shape[-1]))
        batch_size = reshaped_predictions.shape[0]

        soft_predictions = self.softmax(reshaped_predictions)
        loss = self.cross_entropy_loss(soft_predictions, target_index)

        delta = np.zeros_like(soft_predictions)
        delta[np.arange(batch_size), target_index.ravel()] = 1

        dprediction = soft_predictions - delta
        dprediction /= batch_size

        dprediction = dprediction.reshape(predictions.shape)

        return loss, dprediction

    def softmax(self, predictions: np.ndarray):
        assert isinstance(predictions, np.ndarray)

        reshaped = predictions.reshape((-1, predictions.shape[-1]))

        shift = np.max(reshaped, axis=1, keepdims=True)
        shifted_predictions = reshaped - shift
        numerators = np.exp(shifted_predictions)
        denominator = np.sum(numerators, axis=1, keepdims=True)
        probs = numerators / denominator

        probs = probs.reshape(predictions.shape)

        return probs

    def cross_entropy_loss(self, probs: np.ndarray, target_index: np.ndarray):
        assert isinstance(probs, np.ndarray)
        assert isinstance(target_index, np.ndarray)

        reshaped = probs.reshape((-1, probs.shape[-1]))

        batch_size = reshaped.shape[0]
        loss = np.sum(-np.log(reshaped[np.arange(batch_size), target_index.ravel()]))
        loss /= batch_size

        return loss


class MSELoss(Loss):
    def compute(self, predictions: np.ndarray, target: np.ndarray) -> (np.ndarray, np.ndarray):
        assert isinstance(predictions, np.ndarray)
        assert isinstance(target, np.ndarray)
        assert target.shape == predictions.shape
        loss = ((target - predictions) ** 2).mean()
        d_loss = (2 / target.size) * (predictions - target)

        return loss, d_loss
