import numpy as np

from collections import OrderedDict
from layers import FullyConnectedLayer, LstmLayer, Sequential
from module import Module


class NumberSortModule(Module):
    def __init__(self, input_size: int, seq_len: int, hidden_size: int, n_layers: int) -> None:
        super().__init__()
        assert isinstance(input_size, int)
        assert isinstance(seq_len, int)
        assert isinstance(hidden_size, int)
        assert isinstance(n_layers, int)
        assert seq_len >= 1
        assert n_layers >= 1
        assert hidden_size >= 1

        self.input_size = input_size
        self.seq_len: int = seq_len
        self.hidden_size: int = hidden_size
        self.n_layers: int = n_layers

        self._init_layers()

        self.register_module_parameters('lstm_encoder', self.lstm_encoder)
        self.register_module_parameters('lstm_decoder', self.lstm_decoder)
        # self.register_module_parameters('decoder', self.decoder)

    def forward(self, x_input: np.ndarray) -> np.ndarray:
        """forward propagation
        
        Arguments:
            x_input {np.ndarray} -- [description]
        
        Returns:
            np.ndarray -- probabilities of permutation values
        """
        assert x_input.shape[-2] == self.seq_len
        assert x_input.shape[-1] == self.input_size

        # align shape to batch-like (batch_size, seq_len, input_size)
        x_input_aligned = x_input.reshape((-1, self.seq_len, self.input_size))
        batch_size, *_ = x_input_aligned.shape

        predictions = np.empty((batch_size, self.seq_len, self.input_size))

        for sample in range(batch_size):
            self._reload()
            lstm_encoder_out, _ = self.lstm_encoder.forward(x_input_aligned[sample])
            # print("LSTM output", lstm_encoder_out)
            lstm_encoder_out = np.tile(lstm_encoder_out, (self.seq_len, 1))
            _, lstm_decoder_out = self.lstm_decoder.forward(lstm_encoder_out)
            lstm_decoder_out = lstm_decoder_out.reshape(-1, self.seq_len, self.input_size)
            # decoder_out = self.decoder.forward(lstm_encoder_out)
            predictions[sample, :] = lstm_decoder_out

        return predictions.reshape(-1, self.seq_len, self.input_size)

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        # as we use time distributed derivative for decoder
        d_dummy = np.zeros(d_output.shape[1:])[np.newaxis, ...]
        d_decoder_hidden, d_decoder_cell, d_decoder = self.lstm_decoder.backward(d_dummy, d_output)
        *_, d_encoder = self.lstm_encoder.backward(d_decoder[np.newaxis, ...])

        return d_encoder

    def predict(self, x_input: np.ndarray) -> np.ndarray:
        predictions_proba = self.forward(x_input).reshape((-1, self.seq_len, self.input_size))
        predicted_permutation = np.argmax(predictions_proba, axis=-1)

        return predicted_permutation

    def _init_layers(self): # FIXME multi-layer lstm actually not work
        if self.n_layers == 1:
            self.lstm_encoder = LstmLayer(self.input_size, self.hidden_size)
            self.lstm_decoder = LstmLayer(self.hidden_size, self.input_size)
        else:
            lstm_encoders = OrderedDict()
            lstm_encoders['lstm_0'] = LstmLayer(self.input_size, self.hidden_size)
            for i in range(1, self.n_layers):
                lstm_encoders['lstm_' + str(i)] = LstmLayer(self.hidden_size, self.hidden_size)
            self.lstm_encoder = Sequential(lstm_encoders)

            lstm_decoders = OrderedDict()
            lstm_decoders['lstm_0'] = LstmLayer(self.hidden_size, self.hidden_size)
            for i in range(1, self.n_layers):
                lstm_decoders['lstm_' + str(i)] = LstmLayer(self.hidden_size, self.hidden_size)
            self.lstm_decoder = Sequential(lstm_decoders)

        # self.decoder = FullyConnectedLayer(self.hidden_size, self.seq_len * self.seq_len)

    def _reload(self):
        if self.n_layers == 1:
            self.lstm_encoder.reload()
        else:
            for lstm in self.lstm_encoder.modules.values():
                lstm.reload()
