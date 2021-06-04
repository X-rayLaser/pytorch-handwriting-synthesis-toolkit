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

        with h5py.File(self._path, 'r') as f:
            ds_lengths = f['lengths']
            self._num_examples = len(ds_lengths)

    def __len__(self):
        return self._num_examples

    def __getitem__(self, item):
        return load_from_h5(self._path, item)


def load_from_h5(path, index):
    with h5py.File(path, 'r') as f:
        ds_sequences = f['sequences']
        ds_lengths = f['lengths']
        ds_texts = f['texts']

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


class BadStrokeSequenceError(Exception):
    pass
