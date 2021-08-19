import traceback

import torch
import torch.nn as nn
import torch.jit as jit
from typing import List, Tuple
from torch import Tensor
import math
from torch.nn import functional as F
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
error_handler = logging.FileHandler(filename='errors.log')
error_handler.setLevel(logging.ERROR)
logger.addHandler(error_handler)


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
        :param x: tensor of shape (batch_size, 1, input_size)
        :param c: tensor of shape (batch_size, num_characters, alphabet_size)
        :param prev_k: tensor of shape (batch_size, num_components)
        :return: phi (attention weights) of shape (batch_size, 1, num_characters), k of shape (batch_size, num_components)
        """

        x = x[:, 0]

        alpha = torch.exp(self.alpha(x))
        beta = torch.exp(self.beta(x))
        k_new = prev_k + torch.exp(self.k(x))

        batch_size, num_chars, _ = c.shape
        phi = self.compute_attention_weights(alpha, beta, k_new, num_chars)
        return phi, k_new

    def compute_attention_weights(self, alpha, beta, k, char_seq_size: int):
        alpha = alpha.unsqueeze(2).repeat(1, 1, char_seq_size)
        beta = beta.unsqueeze(2).repeat(1, 1, char_seq_size)
        k = k.unsqueeze(2).repeat(1, 1, char_seq_size)
        u = torch.arange(char_seq_size, device=alpha.device)

        densities = alpha * torch.exp(-beta * (k - u) ** 2)
        phi = densities.sum(dim=1).unsqueeze(1)
        return phi

    @staticmethod
    def matmul_3d(phi, c):
        return torch.bmm(phi, c)


class SynthesisNetwork(jit.ScriptModule):
    @classmethod
    def get_default_model(cls, alphabet_size, device, bias=None):
        return cls(3, 400, alphabet_size, device, bias=bias)

    def __init__(self, input_size, hidden_size, alphabet_size, device,
                 gaussian_components=10, output_mixtures=20, bias=None):
        super().__init__()

        self.hidden_size = hidden_size
        self.alphabet_size = alphabet_size
        self.device = device
        self.gaussian_components = gaussian_components

        self.lstm1 = PeepholeLSTM(input_size + alphabet_size, hidden_size)
        self.window = SoftWindow(hidden_size, gaussian_components)
        self.lstm2 = PeepholeLSTM(input_size + hidden_size + alphabet_size, hidden_size)
        self.lstm3 = PeepholeLSTM(input_size + hidden_size + alphabet_size, hidden_size)
        self.mixture = MixtureDensityLayer(hidden_size * 3, output_mixtures, bias)

    @jit.script_method
    def forward(self, x: Tensor, c: Tensor):
        batch_size, steps, _ = x.shape
        hidden1, hidden2, hidden3 = self.get_all_initial_states(batch_size)
        h1, w1 = self.compute_windows(x, c, hidden1)
        (pi, mu, sd, ro, eos), _, _ = self.compute_mixture(x, h1, w1, hidden2, hidden3)
        return (pi, mu, sd, ro), eos

    @jit.script_method
    def compute_windows(self, x: Tensor, c: Tensor, hidden1: Tuple[Tensor, Tensor]):
        batch_size, steps, _ = x.shape
        w_t = self.get_initial_window(batch_size)
        k = torch.zeros(batch_size, self.gaussian_components, device=self.device, dtype=torch.float32)

        h1 = []
        w1 = []

        for t in range(steps):
            x_t = x[:, t:t + 1, :]
            x_with_w = torch.cat([x_t, w_t], dim=-1)
            h_t, hidden1 = self.lstm1(x_with_w, hidden1)
            phi, k = self.window(h_t, c, k)
            w_t = self.window.matmul_3d(phi, c)

            h1.append(h_t)
            w1.append(w_t)

        h = torch.cat(h1, dim=1)
        w = torch.cat(w1, dim=1)
        return h, w

    @jit.script_method
    def compute_mixture(self, x: Tensor, h1: Tensor, w1: Tensor,
                        hidden2: Tuple[Tensor, Tensor], hidden3: Tuple[Tensor, Tensor]):
        inputs = torch.cat([x, h1, w1], dim=-1)
        h2, hidden2 = self.lstm2(inputs, hidden2)

        inputs = torch.cat([x, h2, w1], dim=-1)
        h3, hidden3 = self.lstm3(inputs, hidden3)

        inputs = torch.cat([h1, h2, h3], dim=-1)

        return self.mixture(inputs), hidden2, hidden3

    def get_initial_states(self, batch_size: int):
        h0 = torch.zeros(batch_size, self.hidden_size, device=self.device)
        c0 = torch.zeros_like(h0)
        return h0, c0

    def get_all_initial_states(self, batch_size: int):
        hidden1 = self.get_initial_states(batch_size)
        hidden2 = self.get_initial_states(batch_size)
        hidden3 = self.get_initial_states(batch_size)
        return hidden1, hidden2, hidden3

    def get_initial_input(self):
        return torch.zeros(1, 3, device=self.device)

    def get_initial_window(self, batch_size: int):
        return torch.zeros(batch_size, 1, self.alphabet_size, device=self.device)

    def sample_means(self, context=None, steps=700, stochastic=False):
        outputs, _ = self.sample_means_with_attention(context, steps, stochastic)
        return outputs

    def sample_primed(self, primed_x, c, s, steps=700):
        c = torch.cat([c, s], dim=1)
        c = c.to(self.device)

        batch_size, u, _ = c.shape
        assert batch_size == 1

        primed_x = torch.cat([self.get_initial_input().unsqueeze(0), primed_x], dim=1)
        w = self.get_initial_window(batch_size)
        k = torch.zeros(batch_size, self.gaussian_components, device=self.device, dtype=torch.float32)

        hidden1, hidden2, hidden3 = self.get_all_initial_states(batch_size)

        priming_steps = primed_x.shape[1]

        for t in range(priming_steps):
            x = primed_x[:, t].unsqueeze(1)

            x_with_w = torch.cat([x, w], dim=-1)
            h1, hidden1 = self.lstm1(x_with_w, hidden1)

            phi, k = self.window(h1, c, k)

            w = self.window.matmul_3d(phi, c)

            mixture, hidden2, hidden3 = self.compute_mixture(x, h1, w, hidden2, hidden3)

        states = (hidden1, hidden2, hidden3)
        outputs, _ = self._sample_sequence(x, c, w, k, states, stochastic=True, steps=steps)

        return outputs

    def sample_means_with_attention(self, context=None, steps=700, stochastic=False):
        c = context.to(self.device)
        batch_size, u, _ = c.shape
        assert batch_size == 1
        x = self.get_initial_input().unsqueeze(0)
        w = self.get_initial_window(batch_size)
        k = torch.zeros(batch_size, self.gaussian_components, device=self.device, dtype=torch.float32)

        states = self.get_all_initial_states(batch_size)

        return self._sample_sequence(x, c, w, k, states, stochastic, steps)

    def _sample_sequence(self, x, c, w, k, states, stochastic=True, steps=700):
        hidden1, hidden2, hidden3 = states
        _, u, _ = c.shape

        outputs = []
        attention_weights = []
        for t in range(steps):
            x_with_w = torch.cat([x, w], dim=-1)
            h1, hidden1 = self.lstm1(x_with_w, hidden1)

            phi, k = self.window(h1, c, k)
            attention_weights.append(phi.squeeze(0))

            w = self.window.matmul_3d(phi, c)

            mixture, hidden2, hidden3 = self.compute_mixture(x, h1, w, hidden2, hidden3)
            mixture = self.squeeze(mixture)
            x_temp = self.get_mean_prediction(mixture, stochastic=stochastic).unsqueeze(0)

            if self._is_end_of_string(phi, u, x_temp):
                x_temp[0, 2] = 1.0
                outputs.append(x_temp)
                break

            outputs.append(x_temp)
            x = x_temp.unsqueeze(0)

        return torch.cat(outputs, dim=0), torch.cat(attention_weights, dim=0)

    def _is_end_of_string(self, phi, string_length, x_temp):
        last_phi = phi[0, 0, string_length - 1]
        is_eos = x_temp[0, 2] > 0.5
        return last_phi > 0.8 or (phi[0, 0].argmax() == string_length - 1 and is_eos)

    def squeeze(self, mixture: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]):
        pi, mu, sd, ro, eos = mixture
        return pi[0, 0], mu[0, 0], sd[0, 0], ro[0, 0], eos[0, 0]

    def unsqueeze(self, mixture: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]):
        pi, mu, sd, ro, eos = mixture
        return pi.unsqueeze(0), mu.unsqueeze(0), sd.unsqueeze(0), ro.unsqueeze(0), eos.unsqueeze(0)

    def get_mean_prediction(self, mixture, stochastic=False):
        return get_mean_prediction(mixture, self.device, stochastic)

    def clip_gradients(self, output_clip_value=100, lstm_clip_value=10):

        def output_params():
            for param in self.mixture.parameters():
                yield param

        def lstm_params():
            for param in self.lstm1.parameters():
                yield param

            for param in self.lstm2.parameters():
                yield param

            for param in self.lstm3.parameters():
                yield param

        torch.nn.utils.clip_grad_value_(output_params(), output_clip_value)
        torch.nn.utils.clip_grad_value_(lstm_params(), lstm_clip_value)


class MixtureDensityLayer(nn.Module):
    def __init__(self, input_size, num_components, bias=None):
        super().__init__()

        self.num_components = num_components
        self.pi = nn.Linear(input_size, num_components)
        self.mu = nn.Linear(input_size, num_components * 2)
        self.sd = nn.Linear(input_size, num_components * 2)
        self.ro = nn.Linear(input_size, num_components)
        self.eos = nn.Linear(input_size, 1)

        self.bias = bias

    def forward(self, x):
        pi_hat = self.pi(x)
        sd_hat = self.sd(x)

        if self.bias is None:
            pi = F.softmax(pi_hat, dim=-1)
            sd = torch.exp(sd_hat)
        else:
            pi = F.softmax(pi_hat * (1 + self.bias), dim=-1)
            sd = torch.exp(sd_hat - self.bias)

        mu = self.mu(x)
        ro = torch.tanh(self.ro(x))
        eos = torch.sigmoid(self.eos(x))
        return pi, mu, sd, ro, eos


class HandwritingPredictionNetwork(nn.Module):
    @classmethod
    def get_default_model(cls, device, bias=None):
        return cls(3, 900, 20, device, bias=bias)

    def __init__(self, input_size, hidden_size, num_components, device=None, bias=None):
        super().__init__()
        self.input_size = input_size
        self.lstm = PeepholeLSTM(input_size, hidden_size)
        self.output_layer = MixtureDensityLayer(hidden_size, num_components, bias=bias)

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
        return get_mean_prediction(output, self.device, stochastic)

    def clip_gradients(self, output_clip_value=100, lstm_clip_value=10):
        def output_params():
            for param in self.output_layer.parameters():
                yield param

        def lstm_params():
            for param in self.lstm.parameters():
                yield param

        torch.nn.utils.clip_grad_value_(output_params(), output_clip_value)
        torch.nn.utils.clip_grad_value_(lstm_params(), lstm_clip_value)


def get_mean_prediction(output, device, stochastic):
    pi, mu, sd, ro, eos = output

    num_components = len(pi)
    if stochastic:
        component = torch.multinomial(pi, 1).item()
    else:
        component = pi.cpu().argmax()
    mu1 = mu[component]
    mu2 = mu[component + num_components]

    sd1 = sd[component]
    sd2 = sd[component + num_components]

    component_ro = ro[component]

    x, y = mu1, mu2
    if stochastic:
        try:
            x, y = sample_from_bivariate_mixture(mu1, mu2, sd1, sd2, component_ro)
        except Exception:
            logger.exception('Failed to sample from bi-variate normal distribution:')

    if eos > 0.5:
        eos_flag = 1
    else:
        eos_flag = 0

    return torch.tensor([x, y, eos_flag], device=device)


def sample_from_bivariate_mixture(mu1, mu2, sd1, sd2, ro):
    cov_x_y = ro * sd1 * sd2
    sigma = torch.tensor([[sd1 ** 2, cov_x_y], [cov_x_y, sd2 ** 2]])

    loc = torch.tensor([mu1.item(), mu2.item()])
    gmm = torch.distributions.MultivariateNormal(loc, sigma)

    v = gmm.sample()

    return v[0].item(), v[1].item()


def expand_dims(shape):
    res = 1
    for dim in shape:
        res *= dim
    return res
