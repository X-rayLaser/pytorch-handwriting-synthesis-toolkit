import argparse
import os
from handwriting_synthesis import data
import iam_ondb


class IAMonDBProviderFactory:
    def __init__(self, iam_home, num_examples, train_fraction=0.8):
        db = iam_ondb.IAMonDB(iam_home)
        it = iam_ondb.bounded_iterator(db, num_examples)

        self._num_examples = num_examples
        self._iterator = it.__iter__()

        self._train_size = int(num_examples * train_fraction)
        self._val_size = num_examples - self._train_size

        self._iterated_train = False

    @property
    def train_data_provider(self):
        for example in self._iterate(0, self._train_size):
            yield example

        self._iterated_train = True

    @property
    def val_data_provider(self):
        if not self._iterated_train:
            raise Exception('You have to iterate over training data before iterating over validation data')

        for example in self._iterate(self._train_size, self._num_examples):
            yield example

    def _iterate(self, start, end):
        for i in range(start, end):
            strokes, _, text = next(self._iterator)
            strokes = self._remove_time_components(strokes)
            yield strokes, text

    def _remove_time_components(self, strokes):
        res = []
        for stroke in strokes:
            new_stroke = []
            for x, y, t in stroke:
                new_stroke.append((x, y))
            res.append(new_stroke)
        return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("save_dir", type=str, help="Directory to save training and validation datasets")
    parser.add_argument("num_examples", type=int, help="Training set size")
    parser.add_argument("-l", "--max_len", type=int, default=50, help="Max number of points in a handwriting sequence")
    args = parser.parse_args()

    save_dir = args.save_dir
    num_examples = args.num_examples
    max_len = args.max_len

    train_save_path = os.path.join(save_dir, 'train.h5')
    val_save_path = os.path.join(save_dir, 'val.h5')
    os.makedirs(save_dir, exist_ok=True)

    factory = IAMonDBProviderFactory('../iam_ondb_home', num_examples, max_len)
    train_provider = factory.train_data_provider
    data.build_dataset(train_provider, train_save_path, max_len)

    val_provider = factory.val_data_provider
    data.build_dataset(val_provider, val_save_path, max_len)
