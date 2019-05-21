from module import *


class PlusOperator(Module):
    def __init__(self) -> None:
        super().__init__()
        self.shape: tuple = None

    def forward(self, x1_input: np.ndarray, x2_input: np.ndarray) -> np.ndarray:
        assert x1_input.shape == x2_input.shape
        self.shape: tuple = x1_input.shape

        return x1_input + x2_input

    def backward(self, d_output: np.ndarray) -> (np.ndarray, np.ndarray):
        assert self.shape == d_output.shape

        return d_output, d_output


class MulOperator(Module):
    def __init__(self) -> None:
        super().__init__()
        self.x1_input: np.ndarray = None
        self.x2_input: np.ndarray = None

    def forward(self, x1_input: np.ndarray, x2_input: np.ndarray) -> np.ndarray:
        assert x1_input.shape == x2_input.shape

        self.x1_input: np.ndarray = x1_input
        self.x2_input: np.ndarray = x2_input

        return x1_input * x2_input

    def backward(self, d_output: np.ndarray) -> (np.ndarray, np.ndarray):
        assert d_output.shape == self.x1_input.shape

        return d_output * self.x2_input, d_output * self.x1_input


class RepeatVec(Module):
    def __init__(self, repeats: int) -> None:
        super().__init__()
        assert isinstance(repeats, int)
        self.repeats = repeats
    
    def forward(self, x_input: np.ndarray) -> np.ndarray:
        assert isinstance(x_input, np.ndarray)

        return np.concatenate((x_input, ) * self.repeats, axis=-1)
    
    def backward(self, d_output: np.ndarray) -> np.ndarray:
        assert isinstance(d_output, np.ndarray)
        length = d_output.shape[-1]
        assert length % self.repeats == 0

        return d_output[:, :length // self.repeats] + d_output[:, length // self.repeats:]
