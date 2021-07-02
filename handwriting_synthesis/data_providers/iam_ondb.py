import iam_ondb
from ..data import clean_text
from .base import DataSplittingProvider


class IAMonDBProvider(DataSplittingProvider):
    name = 'iam'

    def __init__(self, training_data_size, validation_data_size=0, iam_home=None):
        if iam_home is None:
            iam_home = '../iam_ondb_home'

        training_data_size, validation_data_size = self._parse_args(training_data_size,
                                                                    validation_data_size)

        iterator = self.get_generator(training_data_size, validation_data_size, iam_home)
        super().__init__(iterator, training_data_size, validation_data_size)

    def _parse_args(self, training_data_size, validation_data_size):
        return int(training_data_size), int(validation_data_size)

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
