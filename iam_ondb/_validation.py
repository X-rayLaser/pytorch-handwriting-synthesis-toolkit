import os


expected_directory_layout = [
    'ascii-all',
    'lineImages-all',
    'lineStrokes-all',
    'original-xml-all',
    'original-xml-part',
    'writers.xml'
]


def validate_dataset(path):
    if not os.path.exists(path):
        raise DatasetNotFoundError()

    if not os.path.isdir(path):
        raise InvalidDatasetError()

    layout = set(os.listdir(path))

    missing = []
    for dir_name in expected_directory_layout:
        if dir_name not in layout:
            missing.append(dir_name)

    if len(missing) > 0:
        raise MissingFilesError(missing)


class DatasetNotFoundError(Exception):
    pass


class InvalidDatasetError(Exception):
    pass


class MissingFilesError(Exception):
    def __init__(self, missing):
        super().__init__()
        self.missing = missing