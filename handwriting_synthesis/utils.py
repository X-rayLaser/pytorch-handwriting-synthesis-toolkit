import re
import os
import traceback

import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from . import data


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

    im = Image.new(mode='L', size=(width, height))

    canvas = ImageDraw.Draw(im)

    if lines:
        for stroke in get_strokes(x_with_offset, y_with_offset, eos):
            canvas.line(stroke, width=10, fill=255)
    else:
        draw_points(x_with_offset, y_with_offset, canvas)
    return im


def draw_points(x, y, canvas):
    for i in range(len(x)):
        xi = x[i]
        yi = y[i]
        canvas.ellipse([(xi, yi), (xi + 5, yi + 5)], width=10, fill=255)


def plot_attention_weights(phi, seq, save_path='img.png'):
    x, y, eos = split_into_components(seq)

    strokes = list(get_strokes(x, y, eos))

    best_phi = phi.cpu().detach().numpy().argmax(axis=1)

    fig, axes = plt.subplots(2)
    fig.set_size_inches(18.5, 10.5)

    axes[0].scatter(x, best_phi)

    c = (1, 0, 0, 1)
    lc = mc.LineCollection(strokes, colors=c, linewidths=2)
    axes[1].add_collection(lc)
    axes[1].autoscale()
    axes[1].invert_yaxis()
    plt.savefig(save_path)
    plt.close('all')


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

    def synthesize(self, c, output_path, show_attention=False):
        try:
            if show_attention:
                sampled_handwriting, phi = self.model.sample_means_with_attention(context=c, steps=self.num_steps,
                                                                                  stochastic=self.stochastic)
                sampled_handwriting = sampled_handwriting.cpu()
                sampled_handwriting = sampled_handwriting * self.sd + self.mu
                plot_attention_weights(phi, sampled_handwriting, output_path)
            else:
                sampled_handwriting = self.model.sample_means(context=c, steps=self.num_steps,
                                                              stochastic=self.stochastic)
                sampled_handwriting = sampled_handwriting.cpu()
                sampled_handwriting = sampled_handwriting * self.sd + self.mu
                visualize_strokes(sampled_handwriting, output_path, lines=True)
        except Exception:
            traceback.print_exc()
