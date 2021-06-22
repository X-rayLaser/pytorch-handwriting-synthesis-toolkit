import unittest

import numpy as np

from handwriting_synthesis import data
from handwriting_synthesis.data import points_stream, flatten_strokes, BadStrokeSequenceError
import shutil
import os


class PointStreamTests(unittest.TestCase):
    def test_point_stream_on_1_stroke_1_point_long(self):
        x = 123
        y = 255
        strokes = [[(x, y)]]
        stream = flatten_strokes(strokes)
        expected = [(x, y, 1)]
        self.assertEqual(expected, stream)

    def test_point_stream_on_1_stroke_3_points_long(self):
        strokes = [[(1, 2), (4, 2), (0, 0)]]
        stream = flatten_strokes(strokes)
        expected = [(1, 2, 0), (4, 2, 0), (0, 0, 1)]
        self.assertEqual(expected, stream)

    def test_point_stream_on_2_strokes_1_point_long(self):
        x1, y1 = 123, 255
        x2, y2 = 999, 888
        strokes = [[(x1, y1)], [(x2, y2)]]
        stream = flatten_strokes(strokes)
        expected = [(x1, y1, 1), (x2, y2, 1)]
        self.assertEqual(expected, stream)

    def test_point_stream_on_2_strokes_2_points_long(self):
        x1, y1 = 123, 255
        x2, y2 = 999, 888
        x3, y3 = 555, 666
        x4, y4 = 222, 333

        strokes = [[(x1, y1), (x2, y2)], [(x3, y3), (x4, y4)]]
        stream = flatten_strokes(strokes)
        expected = [(x1, y1, 0), (x2, y2, 1), (x3, y3, 0), (x4, y4, 1)]
        self.assertEqual(expected, stream)

    def test_flatten_strokes_on_2_strokes_2_points_long(self):
        x1, y1 = 123, 255
        x2, y2 = 999, 888
        x3, y3 = 555, 666
        x4, y4 = 222, 333

        strokes = [[(x1, y1), (x2, y2)], [(x3, y3), (x4, y4)]]
        stream = flatten_strokes(strokes)
        expected = [(x1, y1, 0), (x2, y2, 1), (x3, y3, 0), (x4, y4, 1)]
        self.assertEqual(expected, stream)

    def test_to_offsets_on_empty_sequence(self):
        self.assertEqual([], data.to_offsets([]))

    def test_to_offsets_on_a_single_point(self):
        stream = [(2, 4, 1)]
        offsets = data.to_offsets(stream)
        expected = [(0, 0, 1)]
        self.assertEqual(expected, offsets)

    def test_to_offsets_on_3_points(self):
        stream = [(2, 4, 0), (3, 8, 1), (5, 16, 0)]
        offsets = data.to_offsets(stream)
        expected = [(0, 0, 0), (1, 4, 1), (2, 8, 0)]
        self.assertEqual(expected, offsets)

    def test_to_absolute_coordinates_on_empty_sequence(self):
        self.assertEqual([], data.to_absolute_coordinates([]))

    def test_to_absolute_coordinates_on_1_point_sequence(self):
        offsets = [(1, 4, 1)]
        absolute = data.to_absolute_coordinates(offsets)
        expected = [(1, 4, 1)]
        self.assertEqual(expected, absolute)

    def test_to_absolute_coordinates(self):
        offsets = [(0, 0, 0), (1, 4, 1), (2, 8, 0)]
        absolute = data.to_absolute_coordinates(offsets)
        expected = [(0, 0, 0), (1, 4, 1), (3, 12, 0)]
        self.assertEqual(expected, absolute)

    def test_to_offsets_and_back(self):
        stream = [(0, 0, 0), (2, 4, 0), (3, 8, 1), (5, 16, 0)]
        restored_stream = data.to_absolute_coordinates(data.to_offsets(stream))
        self.assertEqual(stream, restored_stream)

    def test_to_strokes_empty_sequence(self):
        self.assertEqual([], data.to_strokes([]))

    def test_to_strokes_on_1_point_sequence(self):
        points = [(2, 3, 1)]
        strokes = data.to_strokes(points)
        expected = [[(2, 3)]]
        self.assertEqual(expected, strokes)

    def test_to_strokes_when_last_point_is_not_marked_as_end_of_stroke(self):
        points = [(2, 3, 0)]
        strokes = data.to_strokes(points)
        expected = [[(2, 3)]]
        self.assertEqual(expected, strokes)

    def test_to_strokes_on_1_stroke_point_sequence(self):
        points = [(2, 3, 0), (1, 1, 0), (2, 5, 1)]
        strokes = data.to_strokes(points)
        expected = [[(2, 3), (1, 1), (2, 5)]]
        self.assertEqual(expected, strokes)

    def test_to_strokes_on_2_strokes_point_sequence(self):
        points = [(2, 3, 1), (1, 1, 0), (2, 5, 1)]
        strokes = data.to_strokes(points)
        expected = [[(2, 3)], [(1, 1), (2, 5)]]
        self.assertEqual(expected, strokes)

    def test_to_strokes_on_sequence_where_every_point_is_end_of_stroke(self):
        points = [(2, 3, 1), (1, 1, 1), (2, 5, 1)]
        strokes = data.to_strokes(points)
        expected = [[(2, 3)], [(1, 1)], [(2, 5)]]
        self.assertEqual(expected, strokes)

    def test_truncate_offsets_to_zero_size(self):
        points = [(2, 3, 0), (1, 1, 1), (2, 5, 1)]
        self.assertEqual([], data.truncate_sequence(points, 0))

    def test_truncate_offsets_to_size_greater_than_sequence_length(self):
        points = [(2, 3, 0), (1, 1, 1), (2, 5, 1)]
        self.assertEqual(points, data.truncate_sequence(points, 10))

    def test_truncate_offsets_to_size_equal_to_sequence_length(self):
        points = [(2, 3, 0), (1, 1, 1), (2, 5, 1)]
        self.assertEqual(points, data.truncate_sequence(points, len(points)))

    def test_truncate_offsets_after_end_of_stroke(self):
        points = [(2, 3, 0), (1, 1, 1), (2, 5, 1)]
        actual = data.truncate_sequence(points, 2)
        expected = [(2, 3, 0), (1, 1, 1)]
        self.assertEqual(expected, actual)

    def test_truncate_offsets_before_end_of_stroke(self):
        points = [(2, 3, 0), (1, 1, 1), (2, 5, 1)]
        actual = data.truncate_sequence(points, 1)
        expected = [(2, 3, 1)]
        self.assertEqual(expected, actual)

    def test_truncate_should_not_modify_passed_data(self):
        original_points = [(2, 3, 0), (1, 1, 1), (2, 5, 1)]
        points = list(original_points)
        actual = data.truncate_sequence(points, 1)
        expected = [(2, 3, 1)]
        self.assertEqual(original_points, points)


class H5Tests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = os.path.join('test_temp')
        os.makedirs(self.temp_dir, exist_ok=False)
        self.h5_path = os.path.join(self.temp_dir, 'dataset.h5')

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_save_and_load_1_sequence(self):
        points = [[0., 0., 0.], [2., 4., 0.], [-1., 1., 1.]]
        dataset = [(points, 'Hello, world')]
        data.save_to_h5(dataset, self.h5_path, max_length=5)

        h5_dataset = data.H5Dataset(self.h5_path)
        self.assertEqual(1, len(h5_dataset))

        restored_dataset, restored_text = h5_dataset[0]
        self.assertEqual(points, restored_dataset)
        self.assertEqual('Hello, world', restored_text)

        self.assertEqual(5, h5_dataset.max_length)

    def test_save_and_load_2_sequences(self):
        seq1 = [[0., 0., 0.], [2., 4., 0.], [-1., 1., 1.]]
        seq2 = [[0., 0., 0.], [-2., 1., 1.]]

        text1 = 'string 1'
        text2 = 'string 2'

        example1 = (seq1, text1)
        example2 = (seq2, text2)
        dataset = [example1, example2]

        data.save_to_h5(dataset, self.h5_path, max_length=5)

        h5_dataset = data.H5Dataset(self.h5_path)
        self.assertEqual(2, len(h5_dataset))

        self.assertTupleEqual(example1, h5_dataset[0])
        self.assertTupleEqual(example2, h5_dataset[1])

    def test_mu_and_std_for_1_sequence(self):
        seq = [[0., 0., 0.], [2., 4., 0.], [-1., 1., 1.]]

        text1 = 'string 1'

        dataset = [(seq, text1)]

        data.save_to_h5(dataset, self.h5_path, max_length=5)

        expected_mu = tuple(np.array([1 / 3., 5 / 3., 0.], dtype=np.float32))
        std_array = np.array(seq, dtype=np.float32).std(axis=0)
        expected_std = (std_array[0], std_array[1], 1.)

        h5_dataset = data.H5Dataset(self.h5_path)

        self.assertTupleAlmostEqual(expected_mu, h5_dataset.mu, places=6)
        self.assertTupleAlmostEqual(expected_std, h5_dataset.std, places=6)

    def test_mu_and_2_sequences(self):
        seq1 = [[0., 0., 0.], [2., 4., 0.], [-1., 1., 1.]]
        seq2 = [[0., 0., 0.], [-2., 1., 1.]]
        flatten = []
        flatten.extend(seq1)
        flatten.extend(seq2)

        text1 = 'string 1'
        text2 = 'string 2'

        example1 = (seq1, text1)
        example2 = (seq2, text2)
        dataset = [example1, example2]

        data.save_to_h5(dataset, self.h5_path, max_length=5)

        expected_mu = tuple(np.array([- 1 / 5., 6 / 5., 0.], dtype=np.float32))
        std_array = np.array(flatten, dtype=np.float32).std(axis=0)
        expected_std = (std_array[0], std_array[1], 1.)

        h5_dataset = data.H5Dataset(self.h5_path)

        self.assertTupleAlmostEqual(expected_mu, h5_dataset.mu, places=8)
        self.assertTupleAlmostEqual(expected_std, h5_dataset.std, places=8)

    def test_mu_and_std_on_random_sequences(self):
        shape1 = (15, 3)
        shape2 = (12, 3)
        shape3 = (8, 3)
        shape4 = (1, 3)
        min_int = -5
        max_int = 5

        np.random.seed(1)
        a1 = np.random.randint(min_int, max_int, shape1)
        a2 = np.random.randint(min_int, max_int, shape2)
        a3 = np.random.randint(min_int, max_int, shape3)
        a4 = np.random.randint(min_int, max_int, shape4)

        a = np.concatenate([a1, a2, a3, a4])

        sequences = [a1.tolist(), a2.tolist(), a3.tolist(), a4.tolist()]
        texts = ['text1', 'text2', 'text3', 'text4']

        dataset = list(zip(sequences, texts))

        data.save_to_h5(dataset, self.h5_path, max_length=15)

        expected_mu = a.mean(axis=0)
        expected_mu = (expected_mu[0], expected_mu[1], 0.)
        expected_std = a.std(axis=0)
        expected_std = (expected_std[0], expected_std[1], 1.)

        h5_dataset = data.H5Dataset(self.h5_path)
        self.assertTupleAlmostEqual(expected_mu, h5_dataset.mu, places=6)
        self.assertTupleAlmostEqual(expected_std, h5_dataset.std, places=6)

    def assertTupleAlmostEqual(self, t1, t2, places):
        n = len(t1)
        for i in range(n):
            self.assertAlmostEqual(t1[i], t2[i], places)

    #def test_point_stream_on_empty_list(self):
    #    self.assertRaises(BadStrokeSequenceError, lambda: list(points_stream([])))
    #    self.assertRaises(BadStrokeSequenceError, lambda: list(points_stream([[], []])))


class MaxPointsSequenceLengthTests(unittest.TestCase):
    def test_on_single_one_stroke_sequence(self):
        points = [
            [[(3, 5), (4, 4), (1, -1)]]
        ]
        texts = ['']

        data_provider = zip(points, texts)
        expected = 3
        actual = data.get_max_sequence_length(data_provider)
        self.assertEqual(expected, actual)

    def test_on_single_multiple_stroke_sequence(self):
        stroke1 = [(3, 5), (4, 4)]
        stroke2 = [(1, -1)]
        points = [[stroke1, stroke2]]
        texts = ['']

        data_provider = zip(points, texts)
        expected = 3
        actual = data.get_max_sequence_length(data_provider)
        self.assertEqual(expected, actual)

    def test_on_few_sequences(self):
        seq1 = [[(3, 5), (4, 4)], [(1, -1)]]
        seq2 = [[(1, 1)]]
        seq3 = [[(3, 5), (4, 4)], [(1, -1), (12, 12)]]
        points = [seq1, seq2, seq3]
        texts = ['text1', 'text2', 'text3']

        data_provider = zip(points, texts)
        expected = 4
        actual = data.get_max_sequence_length(data_provider)
        self.assertEqual(expected, actual)


class TextCleaningTests(unittest.TestCase):
    def test_can_process_one_apostrophe(self):
        s = "It&apos;s just a text"
        actual = data.clean_text(s)
        expected = "It's just a text"
        self.assertEqual(expected, actual)

    def test_can_process_two_apostrophes(self):
        s = "It&apos;s just a text. And here&apos;s another one."
        actual = data.clean_text(s)
        expected = "It's just a text. And here's another one."
        self.assertEqual(expected, actual)

    def test_can_process_one_quote_code(self):
        s = '&quot;This is a simple unclosed quote.'

        actual = data.clean_text(s)
        expected = '"This is a simple unclosed quote.'
        self.assertEqual(expected, actual)

    def test_can_process_2_quote_codes(self):
        s = 'He said: &quot;something!&quot;'
        actual = data.clean_text(s)
        expected = 'He said: "something!"'
        self.assertEqual(expected, actual)

    def test_can_process_both_apostrophe_and_quotations(self):
        s = 'He said: &quot;That&apos;s something!&quot;'
        actual = data.clean_text(s)
        expected = '''He said: "That's something!"'''
        self.assertEqual(expected, actual)


class TokenizerTests(unittest.TestCase):
    def test_cannot_use_charset_with_non_unique_characters(self):
        self.assertRaises(data.BadCharsetError, lambda: data.Tokenizer(charset='abcda'))

    def test_tokenize_and_detokenize_empty_string(self):
        charset = 'Helo wrd'
        tokenizer = data.Tokenizer(charset)
        tokens = tokenizer.tokenize('')
        self.assertEqual([], tokens)
        s = tokenizer.detokenize(tokens)
        self.assertEqual('', s)

    def test_can_tokenize_using_empty_charset(self):
        s = 'Test'
        tokenizer = data.Tokenizer(charset='')
        self.assertEqual([0] * len(s), tokenizer.tokenize(s))

    def test_can_tokenize_and_detokenize(self):
        charset = 'Helo wrd'
        tokenizer = data.Tokenizer(charset)
        s = 'Hello world'
        tokens = tokenizer.tokenize(s)
        self.assertEqual(len(s), len(tokens))

        reconstructed = tokenizer.detokenize(tokens)
        self.assertEqual(s, reconstructed)

    def test_tokenize_unknown_character(self):
        charset = 'abc'
        tokenizer = data.Tokenizer(charset)

        tokens = tokenizer.tokenize('abcd')
        self.assertEqual([1, 2, 3, 0], tokens)

    def test_tokenizer_returns_correct_charset_size(self):
        self.assertEqual(1, data.Tokenizer(charset='').size)

        charset = 'abc'
        tokenizer = data.Tokenizer(charset)
        self.assertEqual(4, tokenizer.size)

    def test_detokenize_unknown_tokens(self):
        charset = 'a'
        tokenizer = data.Tokenizer(charset)
        actual = tokenizer.detokenize([454, 232, 1, 15])
        expected = 'a'
        self.assertEqual(expected, actual)

    def test_can_work_with_unicode_strings(self):
        text = 'こんにちは'
        charset = str(text)
        tokenizer = data.Tokenizer(charset)
        tokens = tokenizer.tokenize(text)
        self.assertEqual([1, 2, 3, 4, 5], tokens)
        self.assertEqual(text, tokenizer.detokenize(tokens))

    def test_size_should_remain_unchanged(self):
        charset = 'abcd'
        tokenizer = data.Tokenizer(charset)
        expected = 5
        self.assertEqual(expected, tokenizer.size)

        tokens = tokenizer.tokenize('abcdefgi')
        self.assertEqual(expected, tokenizer.size)


class TokenizerSerializationTests(unittest.TestCase):
    def setUp(self):
        dir_path = 'temp_charsets'
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)

        os.makedirs(dir_path, exist_ok=True)
        self.path = os.path.join(dir_path, 'temp_charset.txt')

    def tearDown(self):
        if os.path.isfile(self.path):
            os.remove(self.path)

    def test_can_save_and_load_tokenizer_from_file(self):
        charset = 'abc'
        self.assertCanSaveAndLoad(charset)

    def test_on_nasty_charset(self):
        charset = '''abc~!@#$%^&*()-+=[{}];:'"|\\/.,<>'''
        self.assertCanSaveAndLoad(charset)

    def test_can_save_unicode(self):
        charset = ''.join(set('こんにちは'))
        self.assertCanSaveAndLoad(charset)

    def test_consistency(self):
        charset = 'string'
        data.Tokenizer(charset).save_charset(self.path)

        text = str(charset)
        expected_tokens = [1, 2, 3, 4, 5, 6]

        for i in range(100):
            tokenizer = data.Tokenizer.from_file(self.path)
            tokens = tokenizer.tokenize(text)
            self.assertEqual(expected_tokens, tokens)

    def assertCanSaveAndLoad(self, charset):
        tokenizer = data.Tokenizer(charset)
        self.assertEqual(charset, tokenizer.charset)

        tokenizer.save_charset(self.path)

        tokenizer = data.Tokenizer.from_file(self.path)
        self.assertEqual(charset, tokenizer.charset)


class BuildCharSetTests(unittest.TestCase):
    def test_using_empty_string_generator(self):
        def lines():
            yield ''
            yield ''

        g = lines()
        charset = data.build_charset(g)
        expected = ''
        self.assertEqual(expected, charset)

    def test_using_single_string_generator(self):
        def lines():
            yield 'aabbbcc'

        expected = 'abc'
        charset = data.build_charset(lines())
        self.assertEqual(expected, charset)

    def test_charset_should_be_sorted(self):
        def lines():
            yield 'cacabbzy'

        expected = 'abcyz'
        charset = data.build_charset(lines())
        self.assertEqual(expected, charset)

    def test_using_identical_strings(self):
        def lines():
            yield 'aabbbcc'
            yield 'aabbbcc'

        expected = 'abc'
        charset = data.build_charset(lines())
        self.assertEqual(expected, charset)

    def test_using_multiple_distinct_strings(self):
        def lines():
            yield 'abbab'
            yield 'zabbzct'
            yield 'O'

        expected = 'Oabctz'
        charset = data.build_charset(lines())
        self.assertEqual(expected, charset)

    def test_build_charset_from_unicode_strings(self):
        hello = 'こんにちは'

        def lines():
            yield hello
            yield hello

        expected = ''.join(sorted(hello))
        charset = data.build_charset(lines())
        self.assertEqual(expected, charset)
