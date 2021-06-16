import iam_ondb


class DataProviderFactory:
    def __init__(self, num_examples):
        # todo: accept training_size and validation_size instead
        # todo: pull up the implementation of properties here
        # todo: abstract get_generator method
        self._num_examples = num_examples

    @property
    def train_data_provider(self):
        raise NotImplementedError

    @property
    def val_data_provider(self):
        raise NotImplementedError


class IAMonDBProviderFactory(DataProviderFactory):
    def __init__(self, num_examples, iam_home=None):
        super().__init__(num_examples)

        if iam_home is None:
            iam_home = '../iam_ondb_home'
        train_fraction = 0.8

        db = iam_ondb.IAMonDB(iam_home)
        it = iam_ondb.bounded_iterator(db, num_examples)

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
