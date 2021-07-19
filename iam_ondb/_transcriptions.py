import os
from collections import Counter
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import ParseError

from iam_ondb._utils import InvalidXmlFileError, KwargContainer, file_iterator
from iam_ondb._utils import get_logger


class GeneralInfo:
    def __init__(self, form, capture_time, settings):
        self.Form = form
        self.CaptureTime = capture_time
        self.Settings = settings

    def __str__(self):
        return 'Form: {};\nCaptureTime: {};\nSettings: {}\n'.format(
            self.Form, self.CaptureTime, self.Settings
        )


class Transcription(list):
    def __init__(self):
        super().__init__()

        self.General = GeneralInfo(None, None, None)

    @property
    def text(self):
        lines = [text_line for _, text_line in self]
        return '\n'.join(lines)

    def __str__(self):
        return self.text


def extract_transcription_from_txt_file(path):
    with open(path) as f:
        transcription = Transcription()
        location, file_name = os.path.split(path)
        object_id, _ = os.path.splitext(file_name)

        started = False
        lines = []
        for line in f.readlines():
            if started and line.strip():
                lines.append(line.rstrip())
            if 'CSR:' in line:
                started = True

        for i, line in enumerate(lines):
            transcription.append((f'{object_id}-{i + 1:02}', line))

        return transcription


def extract_transcription(path):
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ParseError:
        raise InvalidXmlFileError()

    transcription = create_transcription_object(root)

    transcription_tags = list(root.iterfind('Transcription'))
    if len(transcription_tags) == 0:
        raise MissingTranscriptionError()

    for tag in transcription_tags:
        for line in tag.iterfind('TextLine'):
            if 'text' in line.attrib and 'id' in line.attrib:
                text = line.attrib['text']
                file_id = line.attrib['id']

                file_id = auto_correct_file_id(file_id, line)
                transcription.append((file_id, text))
    return transcription


def auto_correct_file_id(file_id, line):
    word_tags = list(line.iterfind('Word'))
    candidate_counts = Counter()

    if len(word_tags) > 1:
        for candidate in get_id_candidates(word_tags):
            candidate_counts.update([candidate])

    if len(candidate_counts) > 0:
        file_id, count = candidate_counts.most_common(1)[0]
    return file_id


def get_id_candidates(word_tags):
    for word_tag in word_tags:
        if 'id' in word_tag.attrib:
            word_id = word_tag.attrib['id']
            quadruplet = word_id.split('-')
            file_id = '-'.join(quadruplet[:-1])
            yield file_id


def create_transcription_object(root):
    res = Transcription()
    general = list(root.iterfind('General'))
    if len(general) > 0:
        general_tag = general[0]

        form = create_form_object(general_tag)
        capture = create_capture_object(general_tag)
        setting = create_setting_object(general_tag)

        res.General.Form = form
        res.General.CaptureTime = capture
        res.General.Setting = setting
    return res


def create_form_object(general_tag):
    form_tag = find_tag(general_tag, 'Form')

    if form_tag is None:
        return None

    return parse_tag_attributes(form_tag)


def create_capture_object(general_tag):
    capture_tag = find_tag(general_tag, 'CaptureTime')

    if capture_tag is None:
        return None

    return parse_tag_attributes(capture_tag)


def create_setting_object(general_tag):
    setting_tag = find_tag(general_tag, 'Setting')
    if setting_tag is None:
        return None

    return parse_tag_attributes(setting_tag)


def parse_tag_attributes(tag):
    attributes = dict(tag.attrib)
    return KwargContainer(**attributes)


def find_tag(root, tag_name):
    tags = list(root.iterfind(tag_name))
    if len(tags) > 0:
        return list(root.iterfind(tag_name))[0]


class MissingTranscriptionError(Exception):
    pass


def transcriptions_iterator(transcriptions_dir):
    for path in file_iterator(transcriptions_dir):
        transcription = try_extracting_transcription(path)
        if transcription is not None:
            yield transcription


def try_extracting_transcription(path):
    logger = get_logger()
    logger.info('extracting transcription from {}'.format(path))

    try:
        return extract_transcription(path)
    except InvalidXmlFileError as e:
        logger.error('Invalid xml file: {}'.format(path))
    except MissingTranscriptionError:
        logger.error('"Transcription" tag not found in file: {}'.format(path))
    except:
        logger.exception(
            'Unknown exception raised when getting a transcription object'
        )


def lines_iterator(transcriptions_dir):
    it = transcriptions_iterator(transcriptions_dir)
    for transcription in it:
        for file_id, line in transcription:
            yield file_id, line
