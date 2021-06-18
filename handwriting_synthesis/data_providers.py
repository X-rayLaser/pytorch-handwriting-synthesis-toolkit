import iam_ondb
from .data import clean_text


class Factory:
    @property
    def train_data_provider(self):
        raise NotImplementedError

    @property
    def val_data_provider(self):
        raise NotImplementedError


class DataSplittingFactory(Factory):
    def __init__(self, iterator, training_data_size, validation_data_size=0):
        self._train_size = training_data_size
        self._val_size = validation_data_size
        self._num_examples = self._train_size + self._val_size
        self._iterator = iterator

        self._num_train_yielded = 0
        self._iterated_train = False

    @property
    def train_data_provider(self):
        self._iterated_train = True

        for _ in range(self._train_size):
            yield next(self._iterator)
            self._num_train_yielded += 1

    @property
    def val_data_provider(self):
        if not self._iterated_train:
            raise Exception('You have to iterate over training data before iterating over validation data')

        if self._num_train_yielded != self._train_size:
            raise Exception(f'Not enough examples in iterator. Expected to yield {self._train_size} '
                            f'training examples, but yielded {self._num_train_yielded}.')

        if self._val_size:
            for _ in range(self._val_size):
                yield next(self._iterator)
        else:
            # yield remaining examples
            try:
                while True:
                    yield next(self._iterator)
            except StopIteration:
                print('Stop iteration')


class IAMonDBProviderFactory(DataSplittingFactory):
    def __init__(self, training_data_size, validation_data_size=0, iam_home=None):
        if iam_home is None:
            iam_home = '../iam_ondb_home'

        iterator = self.get_generator(training_data_size, validation_data_size, iam_home)
        super().__init__(iterator, training_data_size, validation_data_size)

    def get_generator(self, training_data_size, validation_data_size, iam_home):
        db = iam_ondb.IAMonDB(iam_home)
        if validation_data_size:
            num_examples = training_data_size + validation_data_size
            it = iam_ondb.bounded_iterator(db, num_examples)
        else:
            it = db.__iter__()

        for strokes, _, text in it:
            strokes = self._remove_time_components(strokes)
            text = clean_text(text)
            yield strokes, text

    def _remove_time_components(self, strokes):
        res = []
        for stroke in strokes:
            new_stroke = []
            for x, y, t in stroke:
                new_stroke.append((x, y))
            res.append(new_stroke)
        return res
