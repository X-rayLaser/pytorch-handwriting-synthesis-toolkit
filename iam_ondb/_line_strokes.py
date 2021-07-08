from xml.etree import ElementTree as ET
from xml.etree.ElementTree import ParseError

from iam_ondb._utils import InvalidXmlFileError, KwargContainer, file_iterator
from iam_ondb._utils import get_logger


class StrokeSet(list):
    def __init__(self):
        super().__init__()
        self.SensorLocation = None
        self.DiagonallyOppositeCoords = None
        self.VerticallyOppositeCoords = None
        self.HorizontallyOppositeCoords = None

    def __str__(self):
        return '\n===================================\n' \
               'StrokeSet\n' \
               'SensorLocation: {};\n' \
               'DiagonallyOppositeCoords: {};\n' \
               'VerticallyOppositeCoords: {};\n' \
               'HorizontallyOppositeCoords: {};\n' \
               'First 5 strokes: {}\n' \
               '===================================\n'.format(self.SensorLocation,
                                                              self.DiagonallyOppositeCoords,
                                                              self.VerticallyOppositeCoords,
                                                              self.HorizontallyOppositeCoords,
                                                              self[:5])


def extract_strokes(path):
    try:
        tree = ET.parse(path)
        root = tree.getroot()

        stroke_set = create_stroke_set(root)

        stroke_set_tags = list(root.iterfind('StrokeSet'))
        if len(stroke_set_tags) == 0:
            raise MissingStrokeSetError()

        for tag in stroke_set_tags:
            for stroke_tag in tag.iterfind('Stroke'):
                stroke_points = make_stroke(stroke_tag)
                if len(stroke_points) > 0:
                    stroke_set.append(stroke_points)
        return stroke_set
    except ParseError:
        raise InvalidXmlFileError()


def create_stroke_set(root):
    stroke_set = StrokeSet()
    white_board_description = list(root.iterfind('WhiteboardDescription'))
    if len(white_board_description) > 0:
        description_tag = white_board_description[0]
        location_tag = list(description_tag.iterfind('SensorLocation'))[0]
        if 'corner' in location_tag.attrib:
            stroke_set.SensorLocation = KwargContainer(**dict(location_tag.attrib))

        stroke_set.DiagonallyOppositeCoords = make_opposite_coords(
            description_tag, 'DiagonallyOppositeCoords')
        stroke_set.VerticallyOppositeCoords = make_opposite_coords(
            description_tag, 'VerticallyOppositeCoords')
        stroke_set.HorizontallyOppositeCoords = make_opposite_coords(
            description_tag, 'HorizontallyOppositeCoords')

    return stroke_set


def make_opposite_coords(description_tag, tag_name):
    from iam_ondb._utils import KwargContainer

    tag = list(description_tag.iterfind(tag_name))[0]
    if 'x' in tag.attrib and 'y' in tag.attrib:
        return KwargContainer(x=int(tag.attrib['x']), y=int(tag.attrib['y']))


def make_stroke(stroke_tag):
    stroke_points = []
    for point in stroke_tag:
        if 'x' in point.attrib and 'y' in point.attrib and 'time' in point.attrib:
            x = int(point.attrib['x'])
            y = int(point.attrib['y'])
            t = float(point.attrib['time'])
            p = (x, y, t)
            stroke_points.append(p)

    return stroke_points


class MissingStrokeSetError(Exception):
    pass


def stroke_sets_iterator(strokes_dir):
    for path in file_iterator(strokes_dir):
        stroke_set = try_extracting_strokes(path)
        if stroke_set is not None:
            yield stroke_set


def try_extracting_strokes(path):
    logger = get_logger()

    try:
        return extract_strokes(path)
    except:
        logger.exception('Unknown error whilst getting a set of strokes')
