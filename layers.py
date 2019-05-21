from collections import OrderedDict

import numpy as np

from typing import Dict
from module import Module, Parameter
from operators import PlusOperator, MulOperator


class FullyConnectedLayer(Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 w_init_parameter: Parameter = None,
                 b_init_parameter: Parameter = None) -> None:
        super().__init__()

        if w_init_parameter:
            assert w_init_parameter.value.shape == (input_size, output_size)
            W = w_init_parameter
        else:
            W = Parameter(0.001 * np.random.randn(input_size, output_size))

        if b_init_parameter:
            assert b_init_parameter.value.shape == (1, output_size)
            B = b_init_parameter
        else:
            B = Parameter(0.001 * np.random.randn(1, output_size))

        self.x_input: np.ndarray = None

        self.register_parameter('W', W)
        self.register_parameter('B', B)

    def forward(self, x_input: np.ndarray) -> np.ndarray:
        self.x_input = x_input
        result = x_input @ self.parameters()['W'].value + self.parameters()['B'].value

        return result

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        self.parameters()['W'].grad = self.parameters()['W'].grad + self.x_input.T @ d_output
        self.parameters()['B'].grad = self.parameters()['B'].grad + np.sum(d_output, axis=0)[np.newaxis, ...]

        d_result = d_output @ self.parameters()['W'].value.T

        return d_result


# FIXME : ugly packing-unpacking
class Sequential(Module):
    def __init__(self, modules: Dict[str, Module]) -> None:
        super().__init__()
        assert isinstance(modules, OrderedDict)
        self.modules = modules
        for name, module in self.modules.items():
            self.register_module_parameters(name, module)

    def forward(self, *x_input: np.ndarray):
        out = x_input
        for module in self.modules.values():
            out = module.forward(*out)
            if not isinstance(out, tuple):
                out = (out,)

        return out[0] if len(out) == 1 else out

    def backward(self, *d_output: np.ndarray):
        df = d_output
        for i, module in enumerate(reversed(self.modules.values())):
            df = module.backward(*df)
            if not isinstance(df, tuple):
                df = (df, )

        return df[0] if len(df) == 1 else df

    def append(self, name: str, module: Module, add_params: bool = True) -> None:
        if name in self.modules.keys():
            raise Exception('Module name already exists')
        self.modules[name] = module
        if add_params:
            self.register_module_parameters(name, module)


class LstmLayer(Module):
    class _LstmTimeStamp(Module):
        def __init__(self, input_size: int, hidden_size: int, prev=None) -> None:
            """
            
            Arguments:
            ----------
                hidden_size {int} -- hidden state size
            
            Keyword Arguments:
            ------------------
                prev {_LstmTimeStamp} -- use weights of previous lstm state
                if None, create weights (default: {None})
            """
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size

            self.d_out_hidden_2: np.ndarray = None

            # print("init")
            if prev is not None:
                # print("prev")
                assert isinstance(prev, type(self))
                self.hidden_state: np.ndarray = prev.hidden_state
                self.cell_state: np.ndarray = prev.cell_state

                self.forget_gate = prev.forget_gate
                self.input_gate_sg = prev.input_gate_sg
                self.input_gate_th = prev.input_gate_th
                self.output_gate = prev.output_gate
            else:
                self.hidden_state: np.ndarray = np.zeros((1, hidden_size))
                self.cell_state: np.ndarray = np.zeros((1, hidden_size))
                # self.hidden_state: np.ndarray = np.zeros((self.input_size, hidden_size))
                # self.cell_state: np.ndarray = np.zeros((self.input_size, hidden_size))

                self.forget_gate: Sequential = Sequential(OrderedDict({
                    'fc': FullyConnectedLayer(self.input_size + hidden_size, hidden_size),
                    'sigmoid': SigmoidLayer()
                }))
                self.input_gate_sg: Sequential = Sequential(OrderedDict({
                    'fc': FullyConnectedLayer(self.input_size + hidden_size, hidden_size),
                    'sigmoid': SigmoidLayer()
                }))
                self.input_gate_th: Sequential = Sequential(OrderedDict({
                    'fc': FullyConnectedLayer(self.input_size + hidden_size, hidden_size),
                    'activate': TanHLayer()  # ReLULayer() #TanHLayer()
                }))
                self.output_gate: Sequential = Sequential(OrderedDict({
                    'fc': FullyConnectedLayer(self.input_size + hidden_size, hidden_size),
                    'sigmoid': SigmoidLayer()
                }))

            # print("self.forget_gate.W :", self.forget_gate.modules['fc'].parameters()['W'].value)

            self.forget_gate_mul: MulOperator = MulOperator()
            self.input_gate_mul: MulOperator = MulOperator()
            self.input_gate_sum: PlusOperator = PlusOperator()

            self.cell_output: TanHLayer = TanHLayer()  # ReLULayer = ReLULayer()#TanHLayer = TanHLayer()
            self.cell_output_mul: MulOperator = MulOperator()

            self.register_module_parameters('forget_gate', self.forget_gate)
            self.register_module_parameters('input_gate_sg', self.input_gate_sg)
            self.register_module_parameters('input_gate_th', self.input_gate_th)
            self.register_module_parameters('output_gate', self.output_gate)

        def forward(self, x_input: np.ndarray) -> np.ndarray:
            """lstm forward propagation
            
            Arguments:
            ----------
                x_input {np.ndarray} -- input value of shape !TODO
            
            Returns:
            --------
                np.ndarray -- result
            """
            assert isinstance(x_input, (np.ndarray))
            assert x_input.shape == (self.input_size,)
            # arr_input = np.array([[x_input]])

            cat: np.ndarray = np.concatenate([self.hidden_state, x_input[np.newaxis, ...]], axis=1)

            forget_gate_out = self.forget_gate.forward(cat)
            forgot_cell = self.forget_gate_mul.forward(self.cell_state, forget_gate_out)

            input_gate_sg_out = self.input_gate_sg.forward(cat)
            input_gate_th_out = self.input_gate_th.forward(cat)
            input_gate_out = self.input_gate_mul.forward(input_gate_sg_out, input_gate_th_out)
            updated_cell = self.input_gate_sum.forward(forgot_cell, input_gate_out)
            updated_cell_tanh = self.cell_output.forward(updated_cell)

            output_gate_out = self.output_gate.forward(cat)
            updated_hidden = self.cell_output_mul.forward(updated_cell_tanh, output_gate_out)

            self.hidden_state = updated_hidden
            self.cell_state = updated_cell

            return updated_hidden

        def backward(self, d_out_hidden: np.ndarray, d_out_cell: np.ndarray = None,
                     d_out_x: np.ndarray = None) -> (np.ndarray, np.ndarray, np.ndarray):
            """back propagation
            
            Arguments:
            ----------
                d_out_hidden {np.ndarray} -- result of next layer back propagation
                d_out_cell {np.ndarray} -- result of future cell_state derivative
                if None, create zeros array (default: {None})
                d_out_hidden_2 {np.ndarray} -- result of next layer back propagation
                used with time distributed sequential model (default: {None})
            
            Returns:
            --------
                (np.ndarray, np.ndarray) -- derivative of hidden and cell states
            """
            assert isinstance(d_out_hidden, np.ndarray)
            assert d_out_hidden.shape == (1, self.hidden_size)
            if self.d_out_hidden_2 is not None:
                assert isinstance(self.d_out_hidden_2, np.ndarray)
                assert self.d_out_hidden_2.shape == (1, self.hidden_size)
                d_out_hidden = d_out_hidden + self.d_out_hidden_2
                self.d_out_hidden_2 = None
            if d_out_cell is not None:
                assert isinstance(d_out_cell, np.ndarray)
                assert d_out_cell.shape == (1, self.hidden_size,)
            else:
                d_out_cell = np.zeros((1, self.hidden_size))

            d_cell_output_mul_cell, d_cell_output_mul_hidden = self.cell_output_mul.backward(d_out_hidden)
            d_result = self.output_gate.backward(d_cell_output_mul_hidden)
            d_cell_output = self.cell_output.backward(d_cell_output_mul_cell)
            d_cell_output = d_cell_output + d_out_cell

            d_forgot_cell, d_input_gate_out = self.input_gate_sum.backward(d_cell_output)

            d_input_gate_sg_out, d_input_gate_th_out = self.input_gate_mul.backward(d_input_gate_out)

            d_result = d_result + self.input_gate_sg.backward(d_input_gate_sg_out)
            d_result = d_result + self.input_gate_th.backward(d_input_gate_th_out)

            d_cell_state, d_forget_gate_out = self.forget_gate_mul.backward(d_forgot_cell)

            d_result = d_result + self.forget_gate.backward(d_forget_gate_out)

            return d_result[:, :self.hidden_size], d_cell_state, d_result[0, self.hidden_size:]

        def set_time_distributed(self, d_out_hidden_2: np.ndarray) -> None:
            assert isinstance(d_out_hidden_2, np.ndarray)
            assert d_out_hidden_2.shape == (1, self.hidden_size)
            self.d_out_hidden_2 = d_out_hidden_2

    def __init__(self, input_size: int, hidden_size: int) -> None:
        """LSTM layer
        
        Arguments:
        ----------
            input_size {int} -- input value size
            hidden_size {int} -- hidden / cell state size
        """
        super().__init__()
        assert isinstance(input_size, int)
        assert isinstance(hidden_size, int)
        assert input_size >= 1
        assert hidden_size >= 1

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cell = self._LstmTimeStamp(input_size, hidden_size)
        self.register_module_parameters('LSTM', self.cell)

        self.history: Sequential = None

    def forward(self, x_input: np.ndarray) -> (np.ndarray, np.ndarray):
        """forward propagation
        
        Arguments:
            x_input {np.ndarray} -- input sequence of shape !TODO
        
        Returns:
            np.ndarray -- predicted output embedding
        """
        assert isinstance(x_input, np.ndarray)
        assert x_input.ndim == 2
        assert x_input.shape[-1] == self.input_size

        future = self._LstmTimeStamp(self.input_size, self.hidden_size, self.cell)
        self.history = Sequential(OrderedDict({}))

        out: np.ndarray = None
        hidden_history = []

        for i, sample in enumerate(x_input):
            out = future.forward(sample)
            hidden_history.append(out)
            past = future
            future = self._LstmTimeStamp(self.input_size, self.hidden_size, past)
            self.history.append(str(i), past)

        return out, np.array(hidden_history)

    def backward(self, d_output: np.ndarray, d_time_distributed: np.ndarray = None) -> np.ndarray:
        assert self.history is not None
        if d_time_distributed is not None:
            for i, timestamp in enumerate(self.history.modules.values()):
                timestamp.set_time_distributed(d_time_distributed[i][np.newaxis, ...])
        d_result = self.history.backward(d_output)
        return d_result

    def reload(self):
        self.step_num = 0
        self.history = None


class DropoutLayer(Module):
    def __init__(self, dropout_chance: float):
        assert isinstance(dropout_chance, float)
        super().__init__()
        self.dropout_chance = float
        self.dropout: np.ndarray = None

    def forward(self, x_input: np.ndarray):
        assert isinstance(x_input, np.ndarray)
        if not self.is_train:
            return x_input
        self.dropout = np.random.rand(*x_input.shape)
        self.dropout[self.dropout < self.dropout_chance] = 0
        self.dropout[self.dropout > 0] = 1 / (1 - self.dropout)
        result = self.dropout * x_input

        return result

    def backward(self, d_output: np.ndarray):
        assert self.dropout is not None, "forward propagation required"
        assert isinstance(d_output, np.ndarray)
        assert d_output.shape == self.dropout.shape

        return self.dropout * d_output


class TanHLayer(Module):
    def __init__(self) -> None:
        super().__init__()
        self.grad: np.ndarray = None

    def forward(self, x_input: np.ndarray) -> np.ndarray:
        assert isinstance(x_input, np.ndarray)

        tanh: np.ndarray = np.tanh(x_input)
        self.grad = 1. - tanh ** 2

        return tanh

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        assert d_output.shape == self.grad.shape

        return d_output * self.grad


class SigmoidLayer(Module):
    def __init__(self) -> None:
        super().__init__()
        self.x_input: np.ndarray = None
        self.grad: np.ndarray = None

    def forward(self, x_input: np.ndarray) -> None:
        assert isinstance(x_input, np.ndarray)

        sigmoid = 1. / (1 + np.exp(-x_input))
        self.grad = sigmoid * (1 - sigmoid)

        return sigmoid

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        assert isinstance(d_output, np.ndarray)
        assert d_output.shape == self.grad.shape

        return d_output * self.grad


class ReLULayer(Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, X):
        self.__grad = np.array(X > 0, dtype=np.float)
        self.__grad[X == 0] = 0.5
        X[X < 0] = 0
        return X

    def backward(self, d_out):
        assert d_out.shape == self.__grad.shape

        d_result = d_out * self.__grad
        return d_result
