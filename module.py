from collections import OrderedDict
from typing import Dict

import numpy as np


class Parameter:
    def __init__(self, value: np.ndarray) -> None:
        self.value: np.ndarray = value
        self.grad:  np.ndarray = np.zeros_like(value)


class Module:
    def __init__(self):
        self.__parameters: Dict[str, Parameter] = OrderedDict()
        self.is_train: bool = False
        pass

    def forward(self, *x_input: np.ndarray):
        raise Exception("Not implemented")

    def backward(self, *d_output: np.ndarray):
        raise Exception("Not implemented")

    def register_parameter(self, param_name: str, parameter: Parameter) -> None:
        if '_Module__parameters' not in self.__dict__:
            raise Exception("Module was not initialized")

        if param_name in self.__parameters.keys():
            raise Exception("Parameter already exists")

        self.__parameters[param_name] = parameter

    def register_module_parameters(self, module_name: str, module) -> None:
        assert isinstance(module, Module)

        for name, parameter in module.parameters().items():
            self.register_parameter(module_name + '_' + name, parameter)
    
    def zero_grad(self):
        for param in self.__parameters.values():
            param.grad = np.zeros_like(param.value)

    def parameters(self) -> Dict[str, Parameter]:
        return self.__parameters
    
    def predict(self, *x_input: np.ndarray):
        raise Exception("Not implemented")

    def train(self) -> None:
        self.is_train = True

    def eval(self) -> None:
        self.is_train = False
