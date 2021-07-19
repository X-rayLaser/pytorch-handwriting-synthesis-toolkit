import os
import tarfile
import logging
from logging import getLogger, FileHandler, Formatter


def extract_archives(dir_path):
    for archive_name in os.listdir(dir_path):
        path = os.path.join(dir_path, archive_name)

        f = tarfile.open(path)
        f.extractall()
        f.close()


class KwargContainer:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        keys_with_values = []

        for k, v in self.__dict__.items():
            if type(v) is str:
                v = '"{}"'.format(v)
            keys_with_values.append('{}={}'.format(k, v))
        s = ', '.join(keys_with_values)
        return 'KwargContainer({})'.format(s)


class InvalidXmlFileError(Exception):
    pass


class ObjectDoesNotExistError(Exception):
    pass


def file_iterator(path):
    for dirpath, dirnames, filenames in os.walk(path):
        for name in filenames:
            yield os.path.join(dirpath, name)


def file_stem_iterator(path):
    for path in file_iterator(path):
        location, file_name = os.path.split(path)
        stem, ext = os.path.splitext(file_name)
        yield stem


_logger = None


def get_logger():
    global _logger
    if _logger:
        return _logger

    logger = getLogger('IAM-onDb')
    handler = FileHandler('IAM-onDb.errors.log')
    formatter = Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s')
    handler.setLevel(logging.ERROR)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    _logger = logger
    return logger


class PathFinder:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def find_path(self, object_id):
        validate_id(object_id)
        dir_candidate = self._get_directory_path(self.root_dir, object_id)

        dir_path = self._choose_path(dir_candidate)
        last_part = self._last_id_part(object_id)

        for file_name in os.listdir(dir_path):
            name_without_extension, _ = os.path.splitext(file_name)
            if self._last_id_part(name_without_extension) == last_part:
                return os.path.join(dir_path, file_name)

        raise ObjectDoesNotExistError()

    def validate_id(self, object_id):
        validate_id(object_id)

    def _choose_path(self, dir_candidate):
        chomped_path = self._chomp_letter(dir_candidate)

        if os.path.isdir(chomped_path):
            return chomped_path
        elif os.path.isdir(dir_candidate):
            return dir_candidate
        else:
            raise ObjectDoesNotExistError()

    def _last_id_part(self, object_id):
        return object_id.split('-')[-1]

    def _get_directory_path(self, root, object_id):
        self.validate_id(object_id)

        parts = object_id.split('-')
        folder = parts[0]
        subfolder = parts[0] + '-' + parts[1]
        return os.path.join(root, folder, subfolder)

    def _chomp_letter(self, s):
        if self._last_is_digit(s):
            return s
        return s[:-1]

    def _last_is_digit(self, s):
        return s[-1].isdigit()


class TranscriptionFinder(PathFinder):
    def find_path(self, object_id):
        dir_candidate = self._get_directory_path(self.root_dir, object_id)
        return self._choose_path(dir_candidate)

    def validate_id(self, object_id):
        parts = object_id.split('-')
        if len(parts) < 2:
            raise MalformedIdError

        validate_parts(parts[:2])


class AsciiFileFinder(TranscriptionFinder):
    def find_path(self, object_id):
        self.validate_id(object_id)
        dir_candidate = self._get_directory_path(self.root_dir, object_id)
        dir_path = self._choose_path(dir_candidate)

        for path in file_iterator(dir_path):
            location, file_name = os.path.split(path)
            stem, ext = os.path.splitext(file_name)
            two_parts = '-'.join(object_id.split('-')[:-1])
            if stem == two_parts:
                return path

        raise ObjectDoesNotExistError()


def validate_id(id_string):
    parts = id_string.split('-')
    if len(parts) != 3:
        raise MalformedIdError(id_string)
    return validate_parts(parts)


def validate_parts(parts):
    for part in parts:
        if not part.isalnum():
            raise MalformedIdError(part)


class MalformedIdError(Exception):
    pass
