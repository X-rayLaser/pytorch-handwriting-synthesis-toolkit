
# make exception classes accessible
from PIL import UnidentifiedImageError
from iam_ondb._validation import DatasetNotFoundError, InvalidDatasetError
from iam_ondb._validation import MissingFilesError
from iam_ondb._utils import InvalidXmlFileError, ObjectDoesNotExistError
from iam_ondb._utils import MalformedIdError
from iam_ondb._line_strokes import MissingStrokeSetError
from iam_ondb._transcriptions import MissingTranscriptionError
from iam_ondb._writers import MissingWritersFileError

# make other entities accessible
from iam_ondb._iam_ondb import IAMonDB, bounded_iterator


# define which classes and functions will be imported after wildcard importing
exception_classes = [
    'UnidentifiedImageError',
    'DatasetNotFoundError',
    'InvalidDatasetError',
    'MissingFilesError',
    'InvalidXmlFileError',
    'ObjectDoesNotExistError',
    'MalformedIdError',
    'MissingStrokeSetError',
    'MissingTranscriptionError',
    'MissingWritersFileError'
]

__all__ = ['IAMonDB', 'bounded_iterator'] + exception_classes
