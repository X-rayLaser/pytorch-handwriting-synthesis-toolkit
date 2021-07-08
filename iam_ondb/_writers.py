import os
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import ParseError

from iam_ondb._utils import KwargContainer, InvalidXmlFileError, get_logger


def extract_writers(path):
    root = try_getting_xml_root(path)

    for writer_tag in list(root.iterfind('Writer')):
        writer = try_parsing_writer_tag(writer_tag)
        if writer is not None:
            yield writer


def try_parsing_writer_tag(writer_tag):
    logger = get_logger()

    try:
        all_attributes = get_attributes(writer_tag)

        children = list(writer_tag.getchildren())
        for element in children:
            parse_inner_element(element, all_attributes)

        return KwargContainer(**all_attributes)
    except:
        logger.exception('Unknown exception raised when parsing writer tag')


def parse_inner_element(element, all_attributes):
    tag_name = element.tag
    if tag_name not in all_attributes:
        all_attributes[tag_name] = []
    all_attributes[tag_name].append(element.text)


def try_getting_xml_root(path):
    if not os.path.isfile(path):
        raise MissingWritersFileError(path)
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ParseError:
        raise InvalidXmlFileError()
    return root


def get_attributes(writer_tag):
    all_attributes = {}
    for k, v in writer_tag.attrib.items():
        all_attributes[k] = [v]
    return all_attributes


class MissingWritersFileError(Exception):
    pass
