from .base import Provider, DataSplittingProvider

# Create a subclasses of Provider here


class MyProvider(Provider):
    name = 'example'

    def get_training_data(self):
        raise NotImplementedError

    def get_validation_data(self):
        raise NotImplementedError
