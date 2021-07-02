def register_class(target_class):
    if target_class.name:
        registry[target_class.name] = target_class


class Meta(type):
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        register_class(cls)
        return cls


class Provider(metaclass=Meta):
    name = ''

    def get_training_data(self):
        raise NotImplementedError

    def get_validation_data(self):
        raise NotImplementedError


class DataSplittingProvider(Provider):
    def __init__(self, iterator, training_data_size, validation_data_size=0):
        self._train_size = training_data_size
        self._val_size = validation_data_size
        self._num_examples = self._train_size + self._val_size
        self._iterator = iterator

        self._num_train_yielded = 0
        self._iterated_train = False

    def get_training_data(self):
        self._iterated_train = True

        for _ in range(self._train_size):
            yield next(self._iterator)
            self._num_train_yielded += 1

    def get_validation_data(self):
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
                pass


registry = {}
