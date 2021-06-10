from collections import defaultdict
import h5py
import numpy as np
import torch.utils.data


def points_stream(strokes):
    for stroke in strokes:
        for x, y in stroke[:-1]:
            yield x, y, 0
        x, y = stroke[-1]
        yield x, y, 1


def flatten_strokes(strokes):
    return list(points_stream(strokes))


def to_strokes(points):
    if not points:
        return []

    strokes = []
    current_stroke = []
    for p in points:
        x, y, eos = p
        current_stroke.append((x, y))
        if eos == 1:
            strokes.append(current_stroke)
            current_stroke = []

    if current_stroke:
        strokes.append(current_stroke)
    return strokes


def to_offsets(points):
    if not points:
        return []

    offsets = []

    first_point = points[0]
    prev = first_point[0], first_point[1]

    for x, y, eos in points:
        prev_x, prev_y = prev
        offsets.append((x - prev_x, y - prev_y, eos))
        prev = x, y

    return offsets


def truncate_sequence(offsets, size):
    offsets = offsets[:size]

    if not offsets:
        return []

    last_one = offsets[-1]
    x, y, _ = last_one
    offsets[-1] = (x, y, 1)
    return offsets


def to_absolute_coordinates(offsets):
    res = []
    prev_x, prev_y = 0, 0

    for offset in offsets:
        x_offset, y_offset, eos = offset
        prev_x += x_offset
        prev_y += y_offset

        res.append((prev_x, prev_y, eos))
    return res


def save_to_h5(data, save_path, max_length):
    with h5py.File(save_path, 'w') as f:
        dt = h5py.string_dtype(encoding='utf-8')
        ds_sequences = f.create_dataset('sequences', (0, max_length, 3), maxshape=(None, max_length, 3))
        ds_lengths = f.create_dataset('lengths', (0,), maxshape=(None,), dtype='i2')
        ds_texts = f.create_dataset('texts', (0,), maxshape=(None,), dtype=dt)
        ds_sequences.attrs['max_length'] = max_length

        for i, (points, text) in enumerate(data):
            a = np.array(points, dtype=np.float16)
            unpadded_length = len(a)
            padding_value = max_length - unpadded_length

            a = np.pad(a, pad_width=[(0, padding_value), (0, 0)])

            ds_sequences.resize((i + 1, max_length, 3))
            ds_sequences[i] = a

            ds_lengths.resize((i + 1,))
            ds_texts.resize((i + 1,))
            ds_lengths[i] = unpadded_length
            ds_texts[i] = text


class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        super().__init__()
        self._path = path
        self._fd = h5py.File(self._path, 'r')
        ds_lengths = self._fd['lengths']
        self._num_examples = len(ds_lengths)

    @property
    def max_length(self):
        ds_sequences = self._fd['sequences']
        return ds_sequences.attrs['max_length']

    def close(self):
        self._fd.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type or exc_val or exc_tb:
            print(f'{exc_type}, {exc_val}\n{exc_tb}')
        self.close()

    def __len__(self):
        return self._num_examples

    def __getitem__(self, item):
        return load_from_h5(self._fd, item)

    @property
    def mu(self):
        # todo: replace with memory efficient implementation
        sequences = self._get_all_points()
        means = np.mean(sequences, axis=0)
        return means[0], means[1], 0.

    @property
    def std(self):
        # todo: replace with memory efficient implementation
        sequences = self._get_all_points()
        stds = np.std(sequences, axis=0)
        return stds[0], stds[1], 1.

    def _get_all_points(self):
        sequences = []
        for i in range(len(self)):
            seq, _ = self[i]
            sequences.extend(seq)
        return sequences


class NormalizedDataset(H5Dataset):
    def __init__(self, path, mu, sd):
        super().__init__(path)
        self._mu = mu
        self._sd = sd

    def __getitem__(self, index):
        points, text = super().__getitem__(index)
        tensor = torch.tensor(points)
        item = self.normalize(tensor)
        return item.numpy().tolist(), text

    @property
    def mu(self):
        return self._mu

    @property
    def std(self):
        return self._sd

    def normalize(self, tensor):
        mu = torch.tensor(self._mu)
        sd = torch.tensor(self._sd)
        return (tensor - mu) / sd

    def denormalize(self, tensor):
        mu = torch.tensor(self._mu)
        sd = torch.tensor(self._sd)
        return tensor * sd + mu


def load_from_h5(fd, index):
    # todo: return sequence as tensor
    ds_sequences = fd['sequences']
    ds_lengths = fd['lengths']
    ds_texts = fd['texts']

    sequence_length = ds_lengths[index]
    sequence = ds_sequences[index][:sequence_length]
    text = ds_texts[index]

    sequence = sequence[:, :].tolist()
    text = text.decode(encoding='utf-8')
    return sequence, text


def preprocess_data(data_provider, max_length):
    for strokes, text in data_provider:
        points = flatten_strokes(strokes)
        offsets = to_offsets(points)
        offsets = truncate_sequence(offsets, max_length)
        yield offsets, text


def build_dataset(data_provider, save_path, max_length):
    generator = preprocess_data(data_provider, max_length)
    save_to_h5(generator, save_path, max_length)


class Tokenizer:
    def __init__(self):
        self.chr_to_int = defaultdict(int)
        self.int_to_chr = {}

        for code in range(ord('A'), ord('Z') + 1):
            token = 1 + code - ord('A')
            ch = chr(code)
            self.chr_to_int[ch] = token
            self.int_to_chr[token] = ch

        for code in range(ord('a'), ord('z') + 1):
            token = 1 + ord('Z') - ord('A') + 1 + code - ord('a')
            ch = chr(code)
            self.chr_to_int[ch] = token
            self.int_to_chr[token] = ch

        self.chr_to_int[' '] = 53
        self.int_to_chr[53] = ' '

    @property
    def size(self):
        num_special_characters = 1
        return len(self.chr_to_int) + num_special_characters

    def tokenize(self, s):
        return [self.chr_to_int[ch] for ch in s]

    def detokenize(self, tokens):
        return ''.join([self.int_to_chr.get(token, '<Unknown_token>') for token in tokens])


class BadStrokeSequenceError(Exception):
    pass
