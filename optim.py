from module import Parameter
from module import Module, Parameter
from typing import Dict

import numpy as np


class Optimizer:
    def __init__(self):
        self.learning_rate: float = None
    
    def step(self):
        raise Exception("Not implemented")
    
    def decay(self, strength: float):
        raise Exception("Not implemented")


class MomentumSGD(Optimizer):
    class _LocalOptim:
        def __init__(self, momentum: float, velocity: float, parameter: Parameter):
            self.momentum = momentum
            self.velocity = velocity
            self.parameter = parameter

    def __init__(self, model: Module, momentum: float = 0, learning_rate: float = 1e-1):
        super().__init__()
        self.learning_rate = learning_rate
        self.localOptimizers: Dict[str, self._LocalOptim] = {}
        for name, param in model.parameters().items():
            self.localOptimizers[name] = self._LocalOptim(momentum, 0, param)
    
    def step(self):
        for optimizer in self.localOptimizers.values():
            self._update(optimizer)

    def _update(self, optimizer):
        assert isinstance(optimizer, self._LocalOptim)
        optimizer.velocity = optimizer.momentum * optimizer.velocity - self.learning_rate * optimizer.parameter.grad
        optimizer.parameter.value = optimizer.parameter.value + optimizer.velocity
    
    def decay(self, strength: float):
        self.learning_rate = strength * self.learning_rate


class Adam(Optimizer):
    class _LocalOptim:
        def __init__(self,
                     velocity: float,
                     accumulated: float,
                     adaptive_learning_rate: float,
                     parameter: Parameter):
            self.velocity = velocity
            self.accumulated = accumulated
            self.adaptive_learning_rate = adaptive_learning_rate
            self.parameter = parameter

    def __init__(self, model: Module, beta1: float = 0.9, beta2: float = 0.999, learning_rate: float = 1e-1):
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.learning_rate = learning_rate
        self.localOptimizers: Dict[str, self._LocalOptim] = {}
        for name, param in model.parameters().items():
            self.localOptimizers[name] = self._LocalOptim(0, 0.001, 0, param)

    def step(self):
        for optimizer in self.localOptimizers.values():
            self._update(optimizer)

    def _update(self, optimizer):
        assert isinstance(optimizer, self._LocalOptim)
        optimizer.velocity = self.beta1 * optimizer.velocity + (1 - self.beta1) * optimizer.parameter.grad
        optimizer.accumulated = self.beta2 * optimizer.accumulated + (1 - self.beta2) * optimizer.parameter.grad ** 2
        optimizer.adaptive_learning_rate = self.learning_rate / np.sqrt(optimizer.accumulated)
        optimizer.parameter.value = optimizer.parameter.value - optimizer.adaptive_learning_rate * optimizer.velocity

    def decay(self, strength: float):
        self.learning_rate = strength * self.learning_rate
