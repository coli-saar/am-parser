from typing import Any, Optional, List

import torch
from allennlp.common import Registrable
from allennlp.common.checks import ConfigurationError
from torch.nn import Module, LSTMCell, Dropout, GRUCell
from allennlp.nn.util import get_dropout_mask


class DecoderCell(Registrable, Module):

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self.hidden_dim

    def get_full_states(self) -> List[Any]:
        """
        Return a full represention of the current state.
        :return: a list of length batch_size
        """
        raise NotImplementedError()

    def set_with_full_states(self, states: List[Any]) -> None:
        """
        Takes the output of get_full_states and restores the internal representation.
        :param states:
        :return:
        """
        raise NotImplementedError()

    def reset_cell(self, batch_size : int, device : Optional[int] = None) -> None:
        """
        Resets the state of the cell.
        @param batch_size:
        @return:
        """
        raise NotImplementedError()

    def set_hidden_state(self, hidden_state : torch.Tensor) -> None:
        """
        Manually set the hidden state, useful to condition on input
        @param hidden_state: shape (batch_size, output_dim)
        @return:
        """
        raise NotImplementedError()

    def get_hidden_state(self) -> torch.Tensor:
        """
        Some representation of the hidden state
        @return: shape (batch_size, output_dim)
        """
        raise NotImplementedError()

    def forward(self, *input: Any, **kwargs: Any):
        raise NotImplementedError("Call step() instead?")

    def step(self, input : torch.Tensor) -> None:
        raise NotImplementedError()


@DecoderCell.register("lstm_cell")
class LSTMCellWrapper(DecoderCell):

    def __init__(self, input_dim: int, hidden_dim: int, num_layers : int = 1, layer_dropout : float = 0.0, recurrent_dropout : float = 0.0):
        super().__init__(input_dim, hidden_dim)
        self.hidden = None
        self.context = None
        assert num_layers >= 1
        self.layers = num_layers
        self._lstm_cell0 = LSTMCell(input_dim, hidden_dim)
        self._lstm_cellL = [LSTMCell(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        self.layer_dropout_rate = layer_dropout
        self.recurrent_dropout_rate = recurrent_dropout

        if self.layer_dropout_rate > 0.0 and num_layers < 2:
            raise ConfigurationError("Layer dropout must be 0.0 if we have only a single layer")

    def reset_cell(self, batch_size : int, device : Optional[int] = None) -> None:
        self.hidden = [torch.zeros(batch_size, self.hidden_dim, device = device) for _ in range(self.layers)]
        self.context = [torch.zeros(batch_size, self.hidden_dim, device = device) for _ in range(self.layers)]

        if self.training and self.layer_dropout_rate > 0.0:
            self.layer_dropout = [get_dropout_mask(self.layer_dropout_rate, self.hidden[i]) for i in range(self.layers)]
        else:
            self.layer_dropout = None

        if self.training and self.recurrent_dropout_rate > 0.0:
            self.recurrent_dropout = [get_dropout_mask(self.layer_dropout_rate, self.hidden[i]) for i in range(self.layers)]
        else:
            self.recurrent_dropout = None

    def set_hidden_state(self, hidden_state : torch.Tensor) -> None:
        assert hidden_state.shape == self.hidden[0].shape
        self.hidden = [hidden_state for _  in range(self.layers)]

    def step(self, input : torch.Tensor) -> None:

        last_hidden = self.hidden[0]
        if self.recurrent_dropout:
            last_hidden = self.recurrent_dropout[0] * last_hidden

        hidden, context = self._lstm_cell0(input, (last_hidden, self.context[0]))
        collected_hidden = [hidden]
        collected_context = [context]

        for layer in range(1, self.layers-1):
            lower_layer = collected_hidden[-1]
            last_hidden = self.hidden[layer]

            if self.layer_dropout:
                lower_layer = self.layer_dropout[layer] * hidden
            if self.recurrent_dropout:
                last_hidden = self.recurrent_dropout[layer] * last_hidden

            hidden, context = self._lstm_cellL[layer](lower_layer,(self.recurrent_dropout[layer] * last_hidden, self.context[layer]))
            collected_hidden.append(hidden)
            collected_context.append(context)

        self.hidden = collected_hidden
        self.context = collected_context

    def get_hidden_state(self) -> torch.Tensor:
        return self.hidden[-1]



@DecoderCell.register("gru_cell")
class GRUCellWrapper(DecoderCell):

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__(input_dim, hidden_dim)
        self.hidden = None
        self.context = None
        self._gru_cell = GRUCell(input_dim, hidden_dim)

    def reset_cell(self, batch_size : int, device : Optional[int] = None) -> None:
        self.hidden = torch.zeros(batch_size, self.hidden_dim, device = device)

    def set_hidden_state(self, hidden_state : torch.Tensor) -> None:
        assert hidden_state.shape == self.hidden.shape
        self.hidden = hidden_state

    def step(self, input : torch.Tensor) -> None:
        self.hidden = self._gru_cell(input, self.hidden)

    def get_hidden_state(self) -> torch.Tensor:
        return self.hidden


@DecoderCell.register("identity")
class IdentityCell(DecoderCell):
    """
    Always return what you receive as input.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__(input_dim, hidden_dim)

    def reset_cell(self, batch_size : int, device : Optional[int] = None) -> None:
        pass

    def set_hidden_state(self, hidden_state : torch.Tensor) -> None:
        pass

    def step(self, input : torch.Tensor) -> None:
        self.input = input

    def get_hidden_state(self) -> torch.Tensor:
        return self.input