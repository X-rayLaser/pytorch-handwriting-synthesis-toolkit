import torch
from torch import nn


class Gate:
    def __init__(self, w_x, w_h, w_c, b):
        self.w_x = w_x
        self.w_h = w_h
        self.w_c = w_c
        self.b = b

    @classmethod
    def random_gate(cls, input_size, hidden_size):
        w_x = torch.rand(input_size, hidden_size)
        w_h = torch.rand(hidden_size, hidden_size)
        w_c = torch.rand(hidden_size,)
        b = torch.rand(hidden_size)
        return cls(w_x, w_h, w_c, b)

    def __call__(self, x, state):
        h, c = state
        return torch.sigmoid(x @ self.w_x + h @ self.w_h + c * self.w_c + self.b)


class CellFormula:
    def __init__(self, w_x, w_h, b):
        self.w_x = w_x
        self.w_h = w_h
        self.b = b

    @classmethod
    def new_formula(cls, input_size, hidden_size):
        w_x = torch.rand(input_size, hidden_size)
        w_h = torch.rand(hidden_size, hidden_size)
        b = torch.rand(hidden_size)
        return cls(w_x, w_h, b)

    def __call__(self, x, i, f, state):
        h, c = state
        return f * c + i * torch.tanh(x @ self.w_x + h @ self.w_h + self.b)


class OutputFormula:
    def __call__(self, o, c):
        return o * torch.tanh(c)


class LSTMCell:
    def __init__(self, input_gate, forget_gate, cell_formula, output_gate):
        self.input_gate = input_gate
        self.forget_gate = forget_gate
        self.cell_formula = cell_formula
        self.output_gate = output_gate
        self.output_formula = OutputFormula()

    @classmethod
    def random_cell(cls, input_size, hidden_size):
        input_gate = Gate.random_gate(input_size, hidden_size)
        forget_gate = Gate.random_gate(input_size, hidden_size)
        cell_formula = CellFormula.new_formula(input_size, hidden_size)
        output_gate = Gate.random_gate(input_size, hidden_size)
        return LSTMCell(input_gate, forget_gate, cell_formula, output_gate)

    def __call__(self, x, state):
        h_prev, c_prev = state
        i = self.input_gate(x, state)
        f = self.forget_gate(x, state)
        c = self.cell_formula(x, i, f, state)
        o = self.output_gate(x, state=(h_prev, c))
        h = self.output_formula(o, c)
        return h, (h, c)


class SlowPeepholeLstm:
    def __init__(self, cell):
        self.cell = cell

    def __call__(self, x, state):
        batch_size, steps, input_size = x.shape

        outputs = []

        for t in range(steps):
            x_t = x[:, t, :]
            h, state = self.cell(x_t, state)
            outputs.append(h.unsqueeze(0))

        y_yat = torch.cat(outputs, dim=0)
        y_hat = y_yat.transpose(0, 1)
        return y_hat, state
