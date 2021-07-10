import re
import os
import traceback
import math
import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from torch.utils.data import DataLoader
from . import data
from .metrics import MovingAverage
from .losses import BiVariateGaussian


class PaddedSequencesBatch:
    def __init__(self, sequences, device=None, padding=0):
        """

        :param sequences: List[List[Tuple]]
        :param padding: value used to pad sequences to be of max_length length
        """
        if not sequences or not sequences[0]:
            raise BadInputError()

        if device is None:
            device = torch.device("cpu")

        self._seqs = sequences

        self._max_len = max([len(s) for s in sequences])

        self._inner_dim = len(sequences[0][0])

        self._batch_size = len(sequences)

        self._tensor = torch.ones(
            self._batch_size, self._max_len, self._inner_dim, dtype=torch.float32, device=device
        ) * padding

        mask = []

        for i in range(self._batch_size):
            seq_len = len(sequences[i])
            mask.extend([True] * seq_len)
            mask.extend([False] * (self._max_len - seq_len))
            for j in range(seq_len):
                self._tensor[i, j] = torch.tensor(sequences[i][j])

        self._mask = torch.tensor(mask, dtype=torch.bool, device=device)

    @property
    def max_length(self):
        return self._max_len

    @property
    def tensor(self):
        return self._tensor

    @property
    def mask(self):
        return self._mask

    def concatenated(self):
        """
        Concatenates all sequences together along dimension 1 while skipping padded values

        :return: Tensor of shape (total_num_steps, inner_dim)
        """
        return self.concatenate_batch(self.tensor)

    def concatenate_batch(self, batch):
        """
        Method that is similar to concatenate, but it works on passed batch

        :param batch: Tensor of self.tensor shape
        :return: Tensor of shape (total_num_steps, inner_dim)
        """

        if batch.shape[0] != self.tensor.shape[0] or batch.shape[1] != self._tensor.shape[1]:
            raise BadInputError(
                f'Expected batch to be of shape {self.tensor.shape}. Got {batch.shape}'
            )

        t = batch.reshape(self._batch_size * self._max_len, -1)
        return t[self.mask]

    def concatenate_predictions(self, y_hat):
        """
        Similar to concatenate_batch, but applies the operation to every tensor
        in the mixture density distribution as well as End-Of-Stroke tensor.

        :param y_hat: tuple consisting of predicted parameters of mixture distribution and predictions for
        End-Of-Stroke (eos_hat) flag.

        Mixture itself is a tuple of 4 tensors: mixture probabilities (pi), means (mu), deviations (sd),
        correlation coefficients (ro).

        pi is a Pytorch 3D Tensor of shape (batch_size, number_of_steps, number_of_mixture_components)
        mu is a Pytorch 3D Tensor of shape (batch_size, number_of_steps, 2 * number_of_mixture_components)
        sd is a Pytorch 3D Tensor of shape (batch_size, number_of_steps, 2 * number_of_mixture_components)
        ro is a Pytorch 3D Tensor of shape (batch_size, number_of_steps, number_of_mixture_components)

        :return: tuple of concatenated mixture tensors and End-Of-Stroke tensor.
        Every tensor now is 2D tensor.

        """
        mixture, eos_hat = y_hat
        pi, mu, sd, ro = mixture

        num_components = pi.shape[-1]
        mu1 = mu[:, :, :num_components]
        mu2 = mu[:, :, num_components:]

        sd1 = sd[:, :, :num_components]
        sd2 = sd[:, :, num_components:]

        pi = self.concatenate_batch(pi)
        mu1 = self.concatenate_batch(mu1)
        mu2 = self.concatenate_batch(mu2)
        mu = torch.cat([mu1, mu2], dim=1)

        sd1 = self.concatenate_batch(sd1)
        sd2 = self.concatenate_batch(sd2)
        sd = torch.cat([sd1, sd2], dim=1)

        ro = self.concatenate_batch(ro)

        eos_hat = self.concatenate_batch(eos_hat)
        packed_output = (pi, mu, sd, ro)
        return packed_output, eos_hat


class BadInputError(Exception):
    pass


def split_into_components(seq):
    seq = seq.cpu().numpy()

    x_offsets = seq[:, 0]
    y_offsets = seq[:, 1]
    eos = seq[:, 2]
    offsets = zip(x_offsets.tolist(), y_offsets.tolist(), eos.tolist())

    x = []
    y = []

    absolute = data.to_absolute_coordinates(offsets)
    for x_abs, y_abs, _ in absolute:
        x.append(x_abs)
        y.append(y_abs)

    return x, y, eos


def visualize_strokes(seq, save_path='img.png', lines=False):
    im = create_strokes_image(seq, lines)
    if im:
        im.save(save_path)


def create_strokes_image(seq, lines=False):
    x, y, eos = split_into_components(seq)
    x = np.array(x)
    y = np.array(y)
    x_with_offset = x - np.floor(x.min())
    y_with_offset = y - np.floor(y.min())

    width = int(x_with_offset.max() + 10)
    height = int(y_with_offset.max() + 10)

    if width * height > 8000 * 2000:
        return

    im = Image.new(mode='L', size=(width, height), color=255)

    canvas = ImageDraw.Draw(im)

    if lines:
        for stroke in get_strokes(x_with_offset, y_with_offset, eos):
            canvas.line(stroke, width=10, fill=0)
    else:
        draw_points(x_with_offset, y_with_offset, canvas)
    return im


def draw_points(x, y, canvas):
    for i in range(len(x)):
        xi = x[i]
        yi = y[i]
        canvas.ellipse([(xi, yi), (xi + 5, yi + 5)], width=10, fill=0)


def plot_attention_weights(phi, seq, save_path='img.png', text=''):
    x, y, eos = split_into_components(seq)

    strokes = list(get_strokes(x, y, eos))

    phi = phi.cpu().detach().numpy()

    fig, axes = plt.subplots(2)
    fig.set_size_inches(18.5, 10.5)

    axes[0].set_facecolor((0, 0, 0))
    y_ticks = list(range(phi.shape[1]))
    if text:
        y_labels = [ch for ch in text]
        axes[0].set_yticks(y_ticks)
        axes[0].set_yticklabels(y_labels, rotation=90)

    for i, single_x in enumerate(x):
        temperatures = phi[i]
        colors = [str(t / temperatures.sum()) for t in temperatures]

        axes[0].scatter([single_x] * len(temperatures), y_ticks, c=colors, s=20, cmap='gray')

    c = (1, 0, 0, 1)
    lc = mc.LineCollection(strokes, colors=c, linewidths=2)
    axes[1].add_collection(lc)
    axes[1].autoscale()
    axes[1].invert_yaxis()
    plt.savefig(save_path)
    plt.close('all')


def plot_mixture_densities(model, norm_mu, norm_sd, save_path, c=None):
    with torch.no_grad():
        _plot_densities(model, norm_mu, norm_sd, save_path, c)


def _plot_densities(model, norm_mu, norm_sd, save_path, c=None):
    seq = model.sample_means(context=c, stochastic=True)

    x0 = model.get_initial_input()
    x = torch.cat([x0.unsqueeze(0), seq.unsqueeze(0)], dim=1)

    if c is not None:
        (pi, mu, sd, ro), eos = model(x, c)
    else:
        (pi, mu, sd, ro), eos = model(x)

    batch_size, num_steps, _ = x.shape
    assert batch_size == 1

    x = x.squeeze(dim=0)

    # revert normalization
    seq = x * norm_sd + norm_mu

    # to absolute_coordinates
    x_hat, y_hat, _ = split_into_components(seq)
    x_hat = np.array(x_hat)
    y_hat = np.array(y_hat)

    min_x = math.floor(x_hat.min())
    min_y = math.floor(y_hat.min())

    max_x = math.ceil(x_hat.max())
    max_y = math.ceil(y_hat.max())

    x_size = max_x - min_x + 1
    y_size = max_y - min_y + 1

    factor = 4

    x_size = int(math.ceil(x_size / factor))
    y_size = int(math.ceil(y_size / factor))

    heatmap = np.zeros((y_size, x_size), dtype=np.float)

    num_components = pi.shape[2]

    deltas = np.indices(heatmap.shape).transpose(1, 2, 0) * factor
    deltas[:, :, 0] += min_y
    deltas[:, :, 1] += min_x

    deltas = torch.tensor(deltas)

    for t in range(1, num_steps):
        x_prev = round(x_hat[t - 1].item())
        y_prev = round(y_hat[t - 1].item())

        deltas_x = deltas[:, :, 1] - x_prev
        deltas_y = deltas[:, :, 0] - y_prev

        deltas_x = (deltas_x - norm_mu[0]) / norm_sd[0]
        deltas_y = (deltas_y - norm_mu[1]) / norm_sd[1]

        pi_t = pi[0, t]
        mu_t = mu[0, t]
        sd_t = sd[0, t]
        ro_t = ro[0, t]

        densities = torch.zeros(heatmap.shape, dtype=torch.float)

        for j in range(num_components):
            mu1 = mu_t[j]
            mu2 = mu_t[num_components + j]

            sd1 = sd_t[j]
            sd2 = sd_t[num_components + j]
            ro_j = ro_t[j]

            d = pi_t[j] * BiVariateGaussian((mu1, mu2), (sd1, sd2), ro_j).density(deltas_x, deltas_y)
            densities += d

        heatmap += densities.numpy()

    figure = plt.figure(figsize=[16, 9], dpi=400)
    plt.imshow(heatmap, cmap='hot')
    plt.colorbar()
    plt.savefig(save_path, dpi=figure.dpi)
    # todo: sliding window approach (compute probabilities only for neighboring patch of previous point)


def get_strokes(x, y, eos):
    assert len(x) == len(y)
    eos_mask = (eos == 1.)
    indices = torch.arange(len(x))
    eos_indices = set(indices[eos_mask].tolist())

    stroke = []
    for i in range(len(x)):
        stroke.append((x[i], y[i]))
        if i in eos_indices:
            yield stroke
            stroke = []

    if stroke:
        yield stroke


def load_saved_weights(model, check_points_dir='check_points'):
    if not os.path.isdir(check_points_dir):
        print(f'Cannot load a model because directory {check_points_dir} does not exist')
        return model, 0

    most_recent = ''
    largest_epoch = 0
    for file_name in os.listdir(check_points_dir):
        matches = re.findall(r'model_at_epoch_([\d]+)', file_name)
        if not matches:
            continue

        iteration_number = int(matches[0])
        if iteration_number > largest_epoch:
            largest_epoch = iteration_number
            most_recent = file_name

    if most_recent:
        recent_checkpoint = os.path.join(check_points_dir, most_recent)
        model.load_state_dict(torch.load(recent_checkpoint))
        print(f'Loaded model weights from {recent_checkpoint} file')
    else:
        print(f'Could not find a model')
    return model, largest_epoch


class HandwritingSynthesizer:
    def __init__(self, model, mu, sd, num_steps, stochastic=True):
        self.model = model
        self.num_steps = num_steps
        self.mu = mu
        self.sd = sd
        self.stochastic = stochastic

    def synthesize(self, c, output_path, show_attention=False, text=''):
        try:
            if show_attention:
                sampled_handwriting, phi = self.model.sample_means_with_attention(context=c, steps=self.num_steps,
                                                                                  stochastic=self.stochastic)
                sampled_handwriting = sampled_handwriting.cpu()
                sampled_handwriting = sampled_handwriting * self.sd + self.mu
                plot_attention_weights(phi, sampled_handwriting, output_path, text=text)
            else:
                sampled_handwriting = self.model.sample_means(context=c, steps=self.num_steps,
                                                              stochastic=self.stochastic)
                sampled_handwriting = sampled_handwriting.cpu()
                sampled_handwriting = sampled_handwriting * self.sd + self.mu
                visualize_strokes(sampled_handwriting, output_path, lines=True)
        except Exception:
            traceback.print_exc()


def get_charset_path_or_raise(charset_path, default_path):
    if charset_path:
        if not os.path.isfile(charset_path):
            raise Exception(
                f'File {charset_path} not found. Charset must be a path to existing text file.'
            )
    else:
        charset_path = default_path
    return charset_path


def collate(batch):
    x = []
    y = []
    for points, text in batch:
        x.append(points)
        y.append(text)
    return x, y


def compute_validation_loss(trainer, dataset, batch_size, verbose=False):
    loader = DataLoader(dataset, batch_size, collate_fn=collate)

    with torch.no_grad():
        ma_loss = MovingAverage()
        ma_loss.reset()

        batches = math.ceil(len(dataset) / batch_size)

        for i, data in enumerate(loader):
            y_hat, loss = trainer.compute_loss(data)
            ma_loss.update(loss)
            if verbose:
                if (i + 1) % 1 == 0:
                    print(f'\rProcessed {i + 1} / {batches} batches', end='')
                if (i + 1) == batches:
                    print()

        return ma_loss.value


def compute_validation_metrics(trainer, dataset, batch_size, metrics, verbose=False):
    loader = DataLoader(dataset, batch_size, collate_fn=collate)

    with torch.no_grad():
        for metric in metrics:
            metric.reset()

        batches = math.ceil(len(dataset) / batch_size)

        for i, data in enumerate(loader):
            y_hat, loss = trainer.compute_loss(data)
            _, eos = y_hat
            points, _ = data
            for metric in metrics:
                ground_true = PaddedSequencesBatch(points, device=eos.device)
                metric.update(y_hat, ground_true)

            if verbose:
                if (i + 1) % 1 == 0:
                    print(f'\rProcessed {i + 1} / {batches} batches', end='')
                if (i + 1) == batches:
                    print()
