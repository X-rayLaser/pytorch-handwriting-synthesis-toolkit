import os

from PIL import UnidentifiedImageError
from iam_ondb._utils import InvalidXmlFileError, ObjectDoesNotExistError

from iam_ondb._line_strokes import extract_strokes, stroke_sets_iterator
from iam_ondb._transcriptions import extract_transcription, transcriptions_iterator, \
    lines_iterator, MissingTranscriptionError, extract_transcription_from_txt_file
from iam_ondb._utils import file_iterator, file_stem_iterator, get_logger
from iam_ondb._utils import PathFinder, TranscriptionFinder, AsciiFileFinder
from iam_ondb._validation import validate_dataset
from iam_ondb._line_images import get_image_data, images_iterator
from iam_ondb._writers import extract_writers


class IAMonDB:
    """
    Simple API providing convenient access to the IAM-OnDB database.

    The class provides a set of methods to retrieve different objects from
    the database such as strokes, images, transcriptions, meta-information, etc.

    # Example

    ```
    db = IAMonDB('path/to/IAMonDB/location')
    for stroke_set, image, line in db:
        print(line)
        image.show()
        break
    ```

    Public methods:
    __iter__

    get_line_examples

    get_image

    get_image_ids

    get_images

    get_line_examples

    get_stroke_set

    get_stroke_set_ids

    get_stroke_sets

    get_text_line

    get_text_line_ids

    get_text_lines

    get_transcriptions

    get_writer

    get_writer_ids

    get_writers
    """
    def __init__(self, db_path):
        r"""Create an instance of IAMonDB class.

        A path should point to the directory that contains the database.

        At the top level, the directory must have a structure as follows:
            ascii-all/
                ascii/
            lineImages-all/
                lineImages/
            lineStrokes-all/
                lineStrokes/
            original-xml-all/
                original/
            original-xml-part/
                original/
            writers.xml

        :param db_path: a path to the directory containing IAM-OnDB database
        :raises DatasetNotFoundError: when database does not exist in a db_path
        :raises InvalidDatasetError: when db_path is a path to a file
        :raises MissingFilesError: when one or more directories are missing
        """
        validate_dataset(db_path)
        self._path = db_path
        self.transcription_dir = os.path.join(db_path, 'original-xml-all', 'original')
        self.images_root = os.path.join(self._path, 'lineImages-all', 'lineImages')
        self.strokes_root = os.path.join(self._path, 'lineStrokes-all', 'lineStrokes')
        self.ascii_dir = os.path.join(self._path, 'ascii-all', 'ascii')
        self.xml_part_dir = os.path.join(self._path, 'original-xml-part', 'original')

        self._stroke_set_ids = None

    def __iter__(self):
        """Iterate over all triplets of a form (stroke_set, image, line).

        If one or more elements in a triplet cannot be retrieved from
        the database, that triplet will be skipped.

        :returns: an iterator returning tuples of (StrokeSet, PIL.Image, str)
        """

        for object_id in self.get_stroke_set_ids():
            example = self._try_getting_example(object_id)
            if example is not None:
                yield example

    def _try_getting_example(self, object_id):
        logger = get_logger()

        try:
            line = self.get_text_line(object_id)
            image_data = self.get_image(object_id)
            stroke_set = self.get_stroke_set(object_id)
            return stroke_set, image_data, line
        except UnidentifiedImageError as e:
            logger.error(repr(e))
        except InvalidXmlFileError:
            logger.error('Failed to parse object with id {}'.format(object_id))
        except ObjectDoesNotExistError as e:
            logger.error(repr(e))
        except:
            logger.exception('Unknown exception raised when trying '
                             'to get an example with id {}'.format(object_id))

    def get_line_examples(self):
        """An alias of __iter__ method"""
        for example in self:
            yield example

    def get_text_lines(self):
        """Iterate over all lines of all transcription texts."""
        for _, line in lines_iterator(self.transcription_dir):
            yield line

    def get_text_line_ids(self):
        """Iterate over ids of all text lines in the database"""
        for line_id, _ in lines_iterator(self.transcription_dir):
            yield line_id

    def get_text_line(self, object_id):
        """Get a line of transcription text by id.

        :param object_id: a string of alphanumeric characters separated by dash
            (for example, d07-470z-01)
        :returns: a string
        :raises ObjectDoesNotExistError: when a line with the id doesn't exist
        :raises MalformedIdError: when id has invalid format
        """
        finder = TranscriptionFinder(self.transcription_dir)
        dir_path = finder.find_path(object_id)

        try:
            for path in file_iterator(dir_path):
                for line_id, line in extract_transcription(path):
                    if line_id == object_id:
                        return line
        except MissingTranscriptionError:
            pass

        finder = AsciiFileFinder(self.ascii_dir)
        path = finder.find_path(object_id)
        for line_id, line in extract_transcription_from_txt_file(path):
            if line_id == object_id:
                return line

        self._raise_object_not_found(object_id)

    def get_transcription_object_by_id(self, object_id):
        finder = TranscriptionFinder(self.transcription_dir)
        dir_path = finder.find_path(object_id)

        for path in file_iterator(dir_path):
            transcription = extract_transcription(path)
            for line_id, line in transcription:
                if line_id == object_id:
                    return transcription

        self._raise_object_not_found(object_id)

    def get_transcriptions(self):
        """Iterate over all transcriptions.

        :returns: an iterator returning instances of Transcription class
        """
        return transcriptions_iterator(self.transcription_dir)

    def get_stroke_sets(self):
        """Iterate over all stroke sets.

        :returns: an iterator returning instances of StrokeSet class
        """
        return stroke_sets_iterator(self.strokes_root)

    def get_stroke_set_ids(self):
        """Iterate over ids of all stroke sets"""
        for name_stem in file_stem_iterator(self.strokes_root):
            stroke_set_id = self._try_getting_stroke_set_id(name_stem)
            if stroke_set_id is not None:
                yield stroke_set_id

    def _try_getting_stroke_set_id(self, name_stem):
        logger = get_logger()
        try:
            self.get_stroke_set(object_id=name_stem)
            return name_stem
        except:
            logger.exception(
                'Unknown error happened when trying to get a stroke set id'
            )

    def get_stroke_set(self, object_id):
        """Get a set of strokes by its id.

        :param object_id: a string of alphanumeric characters separated by dash
            (for example, d07-470z-01)
        :returns: an instance of StrokeSet class
        :raises ObjectDoesNotExistError: when an object with the id doesn't exist
        :raises MalformedIdError: when id has invalid format
        """
        finder = PathFinder(self.strokes_root)
        file_path = finder.find_path(object_id)

        if not os.path.isfile(file_path):
            self._raise_object_not_found(object_id)

        return extract_strokes(file_path)

    def get_images(self):
        """Iterate over all images.

        :returns: an iterator returning instances of PIL.Image class
        """
        return images_iterator(self.images_root)

    def get_image_ids(self):
        """Iterate over ids of all images."""
        for name_stem in file_stem_iterator(self.images_root):
            image_id = self._try_getting_image_id(name_stem)
            if image_id is not None:
                yield image_id

    def _try_getting_image_id(self, name_stem):
        logger = get_logger()
        try:
            self.get_image(name_stem)
            return name_stem
        except:
            logger.exception('Unknown error happened when trying to get image id')

    def get_image(self, object_id):
        """Get an image by its id.

        :param object_id: a string of alphanumeric characters separated by dash
            (for example, d07-470z-01)
        :returns: an instance of PIL.Image class
        :raises ObjectDoesNotExistError: when image with such id doesn't exist
        :raises MalformedIdError: when id has invalid format
        """
        finder = PathFinder(self.images_root)
        file_path = finder.find_path(object_id)

        if not os.path.isfile(file_path):
            raise ObjectDoesNotExistError(file_path)

        return get_image_data(file_path)

    def get_writers(self):
        """Iterate over all objects with information about each writer.

        :returns: an iterator returning instances of KwargContainer class
        """
        writer_path = os.path.join(self._path, 'writers.xml')
        return extract_writers(writer_path)

    def get_writer_ids(self):
        """Iterate over ids of all writers."""
        for writer in self.get_writers():
            name_list = writer.name
            yield name_list[0]

    def get_writer(self, writer_id):
        """Get a writer object by its id.

        :param writer_id: id of a writer (str)
        :returns: instance of KwargContainer
        :raises ObjectDoesNotExistError: when writer with such id doesn't exist
        """
        for writer in self.get_writers():
            if writer.name[0] == str(writer_id):
                return writer

        self._raise_object_not_found(writer_id)

    def get_example_ids_for_writer(self, writer_id):
        # todo: speed this up, add caching
        if not self._stroke_set_ids:
            self._stroke_set_ids = self.get_stroke_set_ids()

        for i, object_id in enumerate(self._stroke_set_ids):
            if i % 500 == 0:
                print(f'processed {i} object ids')
            try:
                transcription = self.get_transcription_object_by_id(object_id)
                form = transcription.General.Form
                if not form:
                    print(f'No form data! {object_id}')
                elif form.writerID == writer_id:
                    yield object_id
            except MissingTranscriptionError:
                print(f'Missing transcription for id {object_id}')
            except ObjectDoesNotExistError:
                print(f'Object with such id was not found: {object_id}')

    def get_first_example_for_writer(self, writer_id):
        gen = self.get_example_ids_for_writer(writer_id)
        object_id = next(gen)
        return self._try_getting_example(object_id)

    def get_all_styles(self):
        pass
        # todo: extract 1 handwriting for every writer
        # todo: save them in a styles folder

    def _raise_object_not_found(self, object_id):
        raise ObjectDoesNotExistError(
            'Object with such id was not found: {}'.format(object_id)
        )


def bounded_iterator(it, stop_index):
    """Create a wrapper to iterate over first few elements of original iterator.

    Iteration may stop earlier when there are no more elements to iterate over.

    :param it: original iterator
    :param stop_index: index of an element at which iteration will stop
    :returns: a wrapped iterator
    """
    for i, element in enumerate(it):
        if i >= stop_index:
            break

        yield element
