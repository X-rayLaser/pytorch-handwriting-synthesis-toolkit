import re
from collections import defaultdict
import h5py
import numpy as np
import torch
import torch.utils.data

from handwriting_synthesis import utils


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

            if (i + 1) % 250 == 0:
                print(f'\rPrepared {i + 1} examples', end='')

    print()
    mu = compute_mu(save_path)
    std = compute_std(mu, save_path)

    with h5py.File(save_path, 'a') as f:
        ds_sequences = f['sequences']
        ds_sequences.attrs['mu'] = mu
        ds_sequences.attrs['std'] = std


def compute_mu(h5_path):
    with h5py.File(h5_path, 'r') as f:
        ds_lengths = f['lengths']
        num_examples = len(ds_lengths)

        s = np.zeros(3, dtype=np.float32)
        n = 0
        for i in range(num_examples):
            sequence, _ = load_from_h5(f, i)
            a = np.array(sequence)
            s += a.sum(axis=0)
            n += len(a)

            if (i + 1) % 250 == 0:
                print(f'\rComputing mean: processed {i + 1} out of {num_examples} examples', end='')
        print('\nComputed mean')

        mu = s / n
        mu[2] = 0.
        return mu


def compute_std(mu, h5_path):
    with h5py.File(h5_path, 'r') as f:
        ds_lengths = f['lengths']
        num_examples = len(ds_lengths)

        squared_sum = np.zeros(3, dtype=np.float32)

        n = 0
        for i in range(num_examples):
            sequence, _ = load_from_h5(f, i)
            a = np.array(sequence)

            spreads = (a - mu) ** 2
            squared_sum += spreads.sum(axis=0)
            n += len(a)

            if (i + 1) % 250 == 0:
                print(f'\rComputing std: processed {i + 1} out of {num_examples} examples', end='')
        print('\nComputed std')

        variance = squared_sum / n
        std = np.sqrt(variance)
        std[2] = 1.
        return std


class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        super().__init__()
        self._path = path
        self._fd = h5py.File(self._path, 'r')
        ds_lengths = self._fd['lengths']
        ds_sequences = self._fd['sequences']

        self._num_examples = len(ds_lengths)
        self._max_len = ds_sequences.attrs['max_length']
        self._means = ds_sequences.attrs['mu']
        self._stds = ds_sequences.attrs['std']

    @property
    def max_length(self):
        return self._max_len

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
        return tuple(self._means)

    @property
    def std(self):
        return tuple(self._stds)

    def _get_all_points(self):
        sequences = []
        for i in range(len(self)):
            seq, _ = self[i]
            sequences.extend(seq)
        return sequences


class NormalizedDataset(H5Dataset):
    def __init__(self, path, mu, sd):
        super().__init__(path)
        # todo: convert to tensors here
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


def get_max_sequence_length(data_provider):
    max_length = 0

    for i, (strokes, _) in enumerate(data_provider):
        points = flatten_strokes(strokes)
        seq_length = len(points)
        max_length = max(max_length, seq_length)

        num_checked = i + 1
        if num_checked % 250 == 0:
            print(f'\rChecked {num_checked} sequences', end='')
    print()
    return max_length


def build_charset(lines_generator):
    charset = set()

    for text in lines_generator:
        charset = charset.union(set(text))

    characters = list(charset)
    sorted_characters = sorted(characters)
    return ''.join(sorted_characters)


def build_and_save_charset(dataset_path, charset_path):
    def gen():
        with H5Dataset(dataset_path) as dataset:
            for i in range(len(dataset)):
                _, text = dataset[i]

                examples_done = i + 1
                if examples_done % 250 == 0:
                    print(
                        f'\rBuilding charset: processed {examples_done} of {len(dataset)} examples',
                        end=''
                    )
                yield text

        print()

    lines_generator = gen()
    charset = build_charset(lines_generator)
    Tokenizer(charset).save_charset(charset_path)


def clean_text(s):
    """
    Substitutes special text codes for a character with an actual character.

    :param s: text string
    :return: text string
    """
    apostrophe = re.compile('&apos;')
    quote = re.compile('&quot;')

    s = apostrophe.sub("'", s)

    return quote.sub('"', s)


def build_dataset(data_provider, save_path, max_length):
    generator = preprocess_data(data_provider, max_length)
    save_to_h5(generator, save_path, max_length)


class Tokenizer:
    @classmethod
    def from_file(cls, path):
        with open(path, 'r') as f:
            s = f.read()
            return cls(s)

    def __init__(self, charset):
        self._validate_charset(charset)
        self._charset = str(charset)
        self._chr_to_int = defaultdict(int)
        self._int_to_chr = {}

        for i, ch in enumerate(self._charset):
            self._int_to_chr[i + 1] = ch
            self._chr_to_int[ch] = i + 1

    def _validate_charset(self, charset):
        collapsed = ''.join(set(charset))
        if len(collapsed) != len(charset):
            raise BadCharsetError(f'Charset has to contain only unique characters: {charset}')

    @property
    def charset(self):
        return str(self._charset)

    @property
    def size(self):
        num_special_characters = 1
        return len(self._charset) + num_special_characters

    def tokenize(self, s):
        return [self._chr_to_int[ch] for ch in s]

    def detokenize(self, tokens):
        return ''.join([self._int_to_chr.get(token, '') for token in tokens])

    def save_charset(self, path):
        with open(path, 'w') as f:
            f.write(self._charset)


class BadCharsetError(Exception):
    pass


class BadStrokeSequenceError(Exception):
    pass


def transcriptions_to_tensor(tokenizer, transcriptions):
    eye = torch.eye(tokenizer.size)

    token_sequences = []
    for s in transcriptions:
        tokens = tokenizer.tokenize(s)
        token_sequences.append(eye[tokens].numpy().tolist())

    batch = utils.PaddedSequencesBatch(token_sequences)
    return batch.tensor
