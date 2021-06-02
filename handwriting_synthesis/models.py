import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.jit as jit
import warnings
from collections import namedtuple
from typing import List, Tuple
from torch import Tensor
import numbers
import time
import math
from torch.nn import functional as F


class PeepholeLSTM(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_ci = nn.Parameter(torch.Tensor(hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        self.W_cf = nn.Parameter(torch.Tensor(hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        self.W_co = nn.Parameter(torch.Tensor(hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self.W_xh = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size * 4))
        self.init_weights()

    def get_initial_state(self, batch_size):
        h0 = torch.zeros(batch_size, self.hidden_size)
        c0 = torch.zeros_like(h0)
        return h0, c0

    def init_weights(self):
        sd = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-sd, sd)

    def set_weights(self, constant):
        for weight in self.parameters():
            weight.data = weight.data * 0 + constant

    @jit.script_method
    def forward(self, x: Tensor, states: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        bs, seq_sz, _ = x.size()
        hidden_seq = torch.jit.annotate(List[Tensor], [])

        h_t, c_t = states

        for t in range(seq_sz):
            x_t = x[:, t, :]

            x_with_h = torch.cat([x_t, h_t], dim=1)
            z = x_with_h @ self.W_xh

            z_i, z_f, z_c, z_o = z.chunk(4, 1)

            i_t = torch.sigmoid(z_i + c_t * self.W_ci + self.b_i)

            f_t = torch.sigmoid(z_f + c_t * self.W_cf + self.b_f)

            c_t = f_t * c_t + i_t * torch.tanh(z_c + self.b_c)

            o_t = torch.sigmoid(z_o + c_t * self.W_co + self.b_o)

            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t)

        hidden_seq = torch.stack(hidden_seq, dim=1)

        return hidden_seq, (h_t, c_t)


class SoftWindow(jit.ScriptModule):
    def __init__(self, input_size, num_components):
        super().__init__()

        self.alpha = nn.Linear(input_size, num_components)
        self.beta = nn.Linear(input_size, num_components)
        self.k = nn.Linear(input_size, num_components)

    @jit.script_method
    def forward(self, x: Tensor, c: Tensor, prev_k: Tensor):
        """
        :param x: tensor of shape (1, input_size)
        :param c: tensor of shape (num_characters, alphabet_size)
        :param prev_k: tensor of shape (num_components,)
        :return: window of shape (1, alphabet_size), k of shape (num_components,)
        """

        x = x[0]

        alpha = torch.exp(self.alpha(x))
        beta = torch.exp(self.beta(x))
        k = prev_k + torch.exp(self.k(x))

        num_chars, _ = c.shape
        densities = []
        for u in range(num_chars):
            t = alpha * torch.exp(-beta * (k - u) ** 2)
            densities.append(t.unsqueeze(0))

        phi = torch.cat(densities, dim=0).sum(dim=1)
        w = torch.matmul(phi, c).unsqueeze(0)
        return w, k


class BatchLessLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = PeepholeLSTM(input_size, hidden_size)

    def forward(self, x: Tensor, initial_states: Tuple[Tensor, Tensor]):
        """

        :param x: tensor of shape (sequence_size, input_size)
        :param initial_states: tensor of shape (hidden_size,)
        :return: x of shape (sequence_size, hidden_size), (h,c) of shape (hidden_size,)
        """
        x = x.unsqueeze(0)
        h, c = initial_states

        h = h.unsqueeze(0)
        c = c.unsqueeze(0)
        initial_states = (h, c)
        y_hat, (h, c) = self.lstm(x, initial_states)
        y_hat = y_hat[0]
        h = h[0]
        c = c[0]
        return y_hat, (h, c)


class BatchLessMixture(nn.Module):
    def __init__(self, input_size, num_components):
        super().__init__()
        self.mixture = MixtureDensityLayer(input_size, num_components)

    def forward(self, x):
        """
        :param x: tensor of shape (sequence_size, input_size)
        :return: (pi, mu, sd, ro, eos)
        """
        x = x.unsqueeze(0)
        pi, mu, sd, ro, eos = self.mixture(x)
        pi = pi[0]
        mu = mu[0]
        sd = sd[0]
        ro = ro[0]
        eos = eos[0]
        return pi, mu, sd, ro, eos


class SynthesisNetwork(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, alphabet_size, device, gaussian_components=10, output_mixtures=20):
        super().__init__()

        self.hidden_size = hidden_size
        self.alphabet_size = alphabet_size
        self.device = device
        self.gaussian_components = gaussian_components

        self.lstm1 = BatchLessLSTM(input_size + alphabet_size, hidden_size)
        self.window = SoftWindow(hidden_size, gaussian_components)
        self.lstm2 = BatchLessLSTM(input_size + hidden_size + alphabet_size, hidden_size)
        self.lstm3 = BatchLessLSTM(input_size + hidden_size + alphabet_size, hidden_size)
        self.mixture = BatchLessMixture(hidden_size * 3, output_mixtures)

        self.hidden1, self.hidden2, self.hidden3 = self.get_all_initial_states()
        self.initial_window = self.get_initial_window()
        self.initial_k = torch.zeros(self.gaussian_components, device=self.device, dtype=torch.float32)

    @jit.script_method
    def forward(self, x: Tensor, c: Tensor):
        batch_size, steps, _ = x.shape
        assert batch_size == 1
        assert c.shape[0] == 1

        x = x[0]
        c = c[0]

        h1, w1 = self.compute_windows(x, c, self.hidden1)
        mixture, _, _ = self.compute_mixture(x, h1, w1, self.hidden2, self.hidden3)
        return self.unsqueeze(mixture)

    @jit.script_method
    def compute_windows(self, x: Tensor, c: Tensor, hidden1: Tuple[Tensor, Tensor]):
        steps, _ = x.shape
        w_t = self.initial_window
        k = self.initial_k

        h1 = []
        w1 = []

        for t in range(steps):
            x_t = x[t:t + 1, :]
            x_with_w = torch.cat([x_t, w_t], dim=1)
            h_t, hidden1 = self.lstm1(x_with_w, hidden1)
            w_t, k = self.window(h_t, c, k)

            h1.append(h_t)
            w1.append(w_t)

        h = torch.cat(h1, dim=0)
        w = torch.cat(w1, dim=0)
        return h, w

    @jit.script_method
    def compute_mixture(self, x: Tensor, h1: Tensor, w1: Tensor, hidden2: Tuple[Tensor, Tensor], hidden3: Tuple[Tensor, Tensor]):
        inputs = torch.cat([x, h1, w1], dim=1)
        h2, hidden2 = self.lstm2(inputs, hidden2)

        inputs = torch.cat([x, h2, w1], dim=1)
        h3, hidden3 = self.lstm3(inputs, hidden3)

        inputs = torch.cat([h1, h2, h3], dim=1)

        return self.mixture(inputs), hidden2, hidden3

    def get_initial_states(self):
        h0 = torch.zeros(self.hidden_size, device=self.device)
        c0 = torch.zeros_like(h0)
        return h0, c0

    def get_all_initial_states(self):
        hidden1 = self.get_initial_states()
        hidden2 = self.get_initial_states()
        hidden3 = self.get_initial_states()
        return hidden1, hidden2, hidden3

    def get_initial_input(self):
        return torch.zeros(1, 3, device=self.device)

    def get_initial_window(self):
        return torch.zeros(1, self.alphabet_size, device=self.device)

    def forward_without_tf(self, x, c):
        steps = x.shape[1]
        batch_size, u, _ = c.shape
        assert batch_size == 1
        c = c[0]
        x = self.get_initial_input()
        w = self.get_initial_window()
        k = torch.zeros(self.gaussian_components, device=self.device, dtype=torch.float32)

        hidden1, hidden2, hidden3 = self.get_all_initial_states()

        pi_list = []
        mu_list = []
        sd_list = []
        ro_list = []
        eos_list = []
        for t in range(steps):
            x_with_w = torch.cat([x, w], dim=1)
            h1, hidden1 = self.lstm1(x_with_w, hidden1)

            w, k = self.window(h1, c, k)

            mixture, hidden2, hidden3 = self.compute_mixture(x, h1, w, hidden2, hidden3)
            pi, mu, sd, ro, eos = mixture
            pi_list.append(pi)
            mu_list.append(mu)
            sd_list.append(sd)
            ro_list.append(ro)
            eos_list.append(eos)

            mixture = self.squeeze(mixture)
            x = self.get_mean_prediction(mixture, stochastic=False)
            x = x.unsqueeze(0)

        pi = torch.cat(pi_list, dim=0)
        mu = torch.cat(mu_list, dim=0)
        sd = torch.cat(sd_list, dim=0)
        ro = torch.cat(ro_list, dim=0)
        eos = torch.cat(eos_list, dim=0)

        mixtures = (pi, mu, sd, ro, eos)
        return self.unsqueeze(mixtures)

    def sample_means(self, context=None, steps=700, stochastic=False):
        c = context
        batch_size, u, _ = c.shape
        assert batch_size == 1
        c = c[0]
        x = self.get_initial_input()
        w = self.get_initial_window()
        k = torch.zeros(self.gaussian_components, device=self.device, dtype=torch.float32)

        hidden1, hidden2, hidden3 = self.get_all_initial_states()

        outputs = []
        for t in range(steps):
            x_with_w = torch.cat([x, w], dim=1)
            h1, hidden1 = self.lstm1(x_with_w, hidden1)

            w, k = self.window(h1, c, k)

            mixture, hidden2, hidden3 = self.compute_mixture(x, h1, w, hidden2, hidden3)
            mixture = self.squeeze(mixture)
            x = self.get_mean_prediction(mixture, stochastic=stochastic)
            x = x.unsqueeze(0)
            outputs.append(x)

        return torch.cat(outputs, dim=0)

    def squeeze(self, mixture: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]):
        pi, mu, sd, ro, eos = mixture
        return pi[0], mu[0], sd[0], ro[0], eos[0]

    def unsqueeze(self, mixture: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]):
        pi, mu, sd, ro, eos = mixture
        return pi.unsqueeze(0), mu.unsqueeze(0), sd.unsqueeze(0), ro.unsqueeze(0), eos.unsqueeze(0)

    def get_mean_prediction(self, mixture, stochastic=False):
        pi, mu, sd, ro, eos = mixture
        num_components = len(pi)
        if stochastic:
            component = torch.multinomial(pi, 1).item()
        else:
            component = pi.cpu().argmax()
        mu1 = mu[component]
        mu2 = mu[component + num_components]

        if eos > 0.5:
            eos_flag = 1
        else:
            eos_flag = 0

        return torch.tensor([mu1, mu2, eos_flag], device=self.device)


class MixtureDensityLayer(nn.Module):
    def __init__(self, input_size, num_components):
        super().__init__()

        self.num_components = num_components
        self.pi = nn.Linear(input_size, num_components)
        self.mu = nn.Linear(input_size, num_components * 2)
        self.sd = nn.Linear(input_size, num_components * 2)
        self.ro = nn.Linear(input_size, num_components)
        self.eos = nn.Linear(input_size, 1)

    def forward(self, x):
        pi = F.softmax(self.pi(x), dim=-1)
        mu = self.mu(x)
        sd = torch.exp(self.sd(x))
        ro = torch.tanh(self.ro(x))
        eos = torch.sigmoid(self.eos(x))
        return pi, mu, sd, ro, eos


class HandwritingPredictionNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_components, device=None):
        super().__init__()
        self.input_size = input_size
        self.lstm = PeepholeLSTM(input_size, hidden_size)
        self.output_layer = MixtureDensityLayer(hidden_size, num_components)

        self._lstm_state = None
        if device is None:
            device = torch.device("cpu")
        self.device = device

    def get_initial_input(self, batch_size):
        return torch.zeros(batch_size, 1, self.input_size, device=self.device)

    def forward(self, x):
        batch_size = len(x)

        if self._lstm_state is None or self._lstm_state[0].shape[0] != batch_size:
            h0, c0 = self.lstm.get_initial_state(batch_size)
            self._lstm_state = (h0.to(self.device), c0.to(self.device))

        x, _ = self.lstm(x, states=self._lstm_state)
        pi, mu, sd, ro, eos = self.output_layer(x)
        return (pi, mu, sd, ro), eos

    def sample_means(self, context=None, steps=700, stochastic=False):
        x = self.get_initial_input(batch_size=1)
        h0, c0 = self.lstm.get_initial_state(batch_size=1)
        hidden = (h0.to(self.device), c0.to(self.device))

        points = torch.zeros(steps, 3, dtype=torch.float32, device=self.device)

        for t in range(steps):
            v, hidden = self.lstm(x, hidden)
            output = self.output_layer(v)
            output = self.unsqueeze(output)
            point = self.get_mean_prediction(output, stochastic=stochastic)
            points[t] = point
            x = point.reshape(1, 1, 3)

        return points

    def unsqueeze(self, t):
        pi, mu, sd, ro, eos = t
        return pi[0, 0], mu[0, 0], sd[0, 0], ro[0, 0], eos[0, 0]

    def get_mean_prediction(self, output, stochastic=False):
        pi, mu, sd, ro, eos = output
        num_components = len(pi)
        if stochastic:
            component = torch.multinomial(pi, 1).item()
        else:
            component = pi.cpu().argmax()
        mu1 = mu[component]
        mu2 = mu[component + num_components]

        if eos > 0.5:
            eos_flag = 1
        else:
            eos_flag = 0

        return torch.tensor([mu1, mu2, eos_flag], device=self.device)



def expand_dims(shape):
    res = 1
    for dim in shape:
        res *= dim
    return res
