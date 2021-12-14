from typing import Optional, List, Any

import torch

import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter

from topdown_parser.nn.decoder_cell import DecoderCell

"""
This is based on code from Ma et al. (2018): 
"""


def VarFastLSTMCellF(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, noise_in=None, noise_hidden=None):
    if noise_in is not None:
        input = input * noise_in

    hx, cx = hidden
    if noise_hidden is not None:
        hx = hx * noise_hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy

class VarRNNCellBase(nn.Module):

    def reset_noise(self, batch_size):
        """
        Should be overriden by all subclasses.
        Args:
            batch_size: (int) batch size of input.
        """
        raise NotImplementedError

class VarFastLSTMCell(VarRNNCellBase):
    """
    A long short-term memory (LSTM) cell with variational dropout.
    .. math::
        \begin{array}{ll}
        i = \mathrm{sigmoid}(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \mathrm{sigmoid}(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hc} h + b_{hg}) \\
        o = \mathrm{sigmoid}(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o * \tanh(c') \\
        \end{array}
    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bias: If `False`, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: True
        p: (p_in, p_hidden) (tuple, optional): the drop probability for input and hidden. Default: (0.5, 0.5)
    Inputs: input, (h_0, c_0)
        - **input** (batch, model_dim): tensor containing input features
        - **h_0** (batch, hidden_size): tensor containing the initial hidden
          state for each element in the batch.
        - **c_0** (batch. hidden_size): tensor containing the initial cell state
          for each element in the batch.
    Outputs: h_1, c_1
        - **h_1** (batch, hidden_size): tensor containing the next hidden state
          for each element in the batch
        - **c_1** (batch, hidden_size): tensor containing the next cell state
          for each element in the batch
    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(4*hidden_size x model_dim)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(4*hidden_size x hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(4*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(4*hidden_size)`
    """

    def __init__(self, input_size, hidden_size, bias=True, p_in=0.5, p_hidden=0.5):
        super(VarFastLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.reset_parameters()
        if p_in < 0 or p_in > 1:
            raise ValueError("input dropout probability has to be between 0 and 1, "
                             "but got {}".format(p_in))
        if p_hidden < 0 or p_hidden > 1:
            raise ValueError("hidden state dropout probability has to be between 0 and 1, "
                             "but got {}".format(p_hidden))
        self.p_in = p_in
        self.p_hidden = p_hidden
        self.noise_in = None
        self.noise_hidden = None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.xavier_uniform_(self.weight_ih)
        if self.bias:
            nn.init.constant_(self.bias_hh, 0.)
            nn.init.constant_(self.bias_ih, 0.)

    def reset_noise(self, batch_size):
        if self.training:
            if self.p_in:
                noise = self.weight_ih.new_empty(batch_size, self.input_size)
                self.noise_in = noise.bernoulli_(1.0 - self.p_in) / (1.0 - self.p_in)
            else:
                self.noise_in = None

            if self.p_hidden:
                noise = self.weight_hh.new_empty(batch_size, self.hidden_size)
                self.noise_hidden = noise.bernoulli_(1.0 - self.p_hidden) / (1.0 - self.p_hidden)
            else:
                self.noise_hidden = None
        else:
            self.noise_in = None
            self.noise_hidden = None

    def forward(self, input, hx):
        return VarFastLSTMCellF(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
            self.noise_in, self.noise_hidden,
        )

@DecoderCell.register("ma-lstm")
class MaLSTMCell(DecoderCell):

    def __init__(self, input_dim: int, hidden_dim: int, input_dropout : float = 0.0, recurrent_dropout : float = 0.0):
        super().__init__(input_dim, hidden_dim)
        self.lstm_cell = VarFastLSTMCell(input_dim, hidden_dim, True, input_dropout, recurrent_dropout)

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self.hidden_dim

    def reset_cell(self, batch_size : int, device : Optional[int] = None) -> None:
        """
        Resets the state of the cell.
        @param batch_size:
        @return:
        """
        self.lstm_cell.reset_noise(batch_size)
        self.batch_size = batch_size
        self.hidden = torch.zeros(batch_size, self.hidden_dim, device = device)
        self.context = torch.zeros(batch_size, self.hidden_dim, device = device)

    def set_hidden_state(self, hidden_state : torch.Tensor) -> None:
        """
        Manually set the hidden state, useful to condition on input
        @param hidden_state: shape (batch_size, output_dim)
        @return:
        """
        self.hidden = hidden_state

    def get_hidden_state(self) -> torch.Tensor:
        """
        Some representation of the hidden state
        @return: shape (batch_size, output_dim)
        """
        return self.hidden


    def step(self, input : torch.Tensor) -> None:
        self.hidden, self.context = self.lstm_cell.forward(input, (self.hidden, self.context))


    def get_full_states(self) -> List[Any]:
        """
        Return a full represention of the current state.
        :return: a list of length batch_size
        """
        return [(self.hidden[i], self.context[i]) for i in range(self.batch_size)]

    def set_with_full_states(self, states: List[Any]) -> None:
        """
        Takes the output of get_full_states and restores the internal representation.
        :param states:
        :return:
        """
        hidden = []
        context = []
        for h,c in states:
            hidden.append(h)
            context.append(c)
        self.hidden = torch.stack(hidden, dim=0)
        self.context = torch.stack(context, dim=0)