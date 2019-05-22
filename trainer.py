from copy import deepcopy

import numpy as np
from metrics import multiclass_accuracy
from module import Module
from optim import Optimizer
from loss import *
import random

class Dataset:
    def __init__(self,
                 train_X: np.ndarray,
                 train_y: np.ndarray,
                 val_X: np.ndarray,
                 val_y: np.ndarray):
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y


# FIXME: not general, but works fine with NumberSorter
class Trainer:
    def __init__(self,
                 model: Module,
                #  loss: Loss,
                 dataset: Dataset,
                 optimizer: Optimizer,
                 num_epochs: int = 20,
                 batch_size: int = 20,
                 learning_rate_decay: float = 1.0,
                 loss = 'mse') -> None:
        assert isinstance(model, Module)
        assert isinstance(loss, str)
        # assert isinstance(loss, Loss)
        assert isinstance(dataset, Dataset)
        assert isinstance(optimizer, Optimizer)
        assert isinstance(num_epochs, int)
        assert isinstance(batch_size, int)
        assert isinstance(learning_rate_decay, float)

        self.model:         Module = model
        self.dataset:       Dataset = dataset
        self.loss:          str = loss
        # self.loss:          Loss = loss
        self.optimizer:     Optimizer = optimizer
        self.num_epochs:    int = num_epochs
        self.batch_size:    int = batch_size
        self.learning_rate_decay: float = learning_rate_decay

        self.optimizers = None

    def compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        indices = np.arange(X.shape[0])
        sections = np.arange(self.batch_size, X.shape[0], self.batch_size)
        batches_indices = np.array_split(indices, sections)

        y = y.argmax(axis=-1)

        pred = np.zeros_like(y)

        for batch_indices in batches_indices:
            batch_X = X[batch_indices]
            pred_batch = self.model.predict(batch_X)
            pred[batch_indices] = pred_batch

        return multiclass_accuracy(pred, y)

    def fit(self, print_log: bool = True) -> (float, float, float):
        num_train = self.dataset.train_X.shape[0]

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for epoch in range(self.num_epochs):
            shuffled_indices = np.arange(num_train)
            # np.random.shuffle(shuffled_indices)
            sections = np.arange(self.batch_size, num_train, self.batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            batch_losses = []

            for batch_indices in batches_indices:
                self.model.zero_grad()
                batch = self.dataset.train_X[batch_indices]
                target_index = self.dataset.train_y[batch_indices]
                batch_loss = 0
                d_loss_value_accum = 0
                for i, sample in enumerate(batch):
                    pred = self.model.forward(sample)
                    pred = pred.reshape(pred.shape[-2], pred.shape[-1])
                    if self.loss == 'mse':
                        loss_value, d_loss_value = MSELoss().compute(pred, target_index[i])
                    else:
                        classes_num = target_index[i].shape[-1]
                        target_indices = target_index[i].argmax(axis=-1)
                        pred = pred.reshape(-1, classes_num)
                        loss_value, d_loss_value = CrossEntropy().compute(pred, target_indices)
                    d_loss_value_accum = d_loss_value_accum + d_loss_value
                    batch_loss += loss_value
                d_loss_value_accum = d_loss_value_accum / len(batch)
                self.model.backward(d_loss_value_accum)
                batch_loss = batch_loss / len(batch)
                self.optimizer.step()

                batch_losses.append(batch_loss)

            if np.not_equal(self.learning_rate_decay, 1.0):
                self.optimizer.decay(self.learning_rate_decay)

            avg_loss = np.mean(batch_losses)

            train_accuracy = self.compute_accuracy(self.dataset.train_X,
                                                   self.dataset.train_y)

            val_accuracy = self.compute_accuracy(self.dataset.val_X,
                                                 self.dataset.val_y)

            if print_log:
                print("Loss: %f, Train accuracy: %f, val accuracy: %f" %
                      (avg_loss, train_accuracy, val_accuracy))

                # rand_index = random.randint(0, self.dataset.val_y.shape[0] - 1)
                # testX = self.dataset.val_X[rand_index]
                # testY = self.dataset.val_y[rand_index]
                # test_num = testX.argmax(axis=-1).ravel()
                # srtd = np.eye(testX.shape[-1])[np.sort(test_num)]
                # predicted = self.model.predict(testX).ravel()
                # predicted_forward = self.model.forward(testX)
                # predicted_forward = predicted_forward.reshape(predicted_forward.shape[-2], predicted_forward.shape[-1])
                # predicted_onehot = np.eye(testX.shape[-1])[predicted_forward.argmax(axis=-1)]
                # print("unsorted: ", test_num)
                # print("sorted:   ", np.sort(test_num))
                # print("predicted:", predicted)

                # print('actual onehot', srtd, sep='\n')
                # print('label        ', testY, sep='\n')
                # print('predicted', predicted_forward, sep='\n')
                # onehotdiff = predicted_forward - testY
                # onehotdiff[onehotdiff < 0] = -1
                # onehotdiff[onehotdiff > 0] = 1
                # print("onehot diff:", onehotdiff, sep='\n')
                # loss, d_loss = self.loss.compute(predicted_forward, testY)
                # print("d_loss:", d_loss, sep='\n')
                # d_loss[d_loss < 0] = -1
                # d_loss[d_loss > 0] = 1
                # print("d_loss:", d_loss, sep='\n')

            loss_history.append(avg_loss)
            train_acc_history.append(train_accuracy)
            val_acc_history.append(val_accuracy)

        return loss_history, train_acc_history, val_acc_history
