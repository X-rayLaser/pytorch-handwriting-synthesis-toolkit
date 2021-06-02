import unittest
import torch
import utils


class PaddedSequencesBatchTests(unittest.TestCase):
    def setUp(self):
        s = [[[3], [4]]]
        self.single_seq_batch = utils.PaddedSequencesBatch(s)

        s = [[[1]], [[4], [4], [4]], [[2], [4]]]
        self.few_seqs_batch = utils.PaddedSequencesBatch(s)

    def test_should_contain_at_least_1_sequence(self):
        self.assertRaises(Exception, lambda: utils.PaddedSequencesBatch([]))
        self.assertRaises(Exception, lambda: utils.PaddedSequencesBatch([[]]))

    def test_max_length_on_single_sequence(self):
        self.assertEqual(self.single_seq_batch.max_length, 2)

    def test_max_length_on_3_sequences(self):
        self.assertEqual(self.few_seqs_batch.max_length, 3)

    def test_tensor_on_single_sequence(self):
        expected_tensor = torch.tensor([
            [[3], [4.]]
        ])
        self.assertEqual(self.single_seq_batch.tensor.shape, expected_tensor.shape)
        self.assertTrue(torch.allclose(self.single_seq_batch.tensor, expected_tensor))

    def test_tensor_on_3_sequences(self):
        expected_tensor = torch.tensor([
            [[1], [0], [0.]],
            [[4], [4], [4]],
            [[2], [4], [0]]
        ])

        self.assertEqual(self.few_seqs_batch.tensor.shape, expected_tensor.shape)
        self.assertTrue(torch.allclose(self.few_seqs_batch.tensor, expected_tensor))

    def test_mask_on_single_sequence(self):
        expected_mask = torch.tensor([True, True])
        self.assertTrue(torch.all(self.single_seq_batch.mask == expected_mask))

    def test_mask_on_3_sequences(self):
        expected_mask = torch.tensor([True, False, False, True, True, True, True, True, False])
        self.assertTrue(torch.all(self.few_seqs_batch.mask == expected_mask))

    def test_concatenated_on_single_sequence(self):
        tensor = self.single_seq_batch.concatenated()
        expected_tensor = torch.tensor([[3.], [4.]])
        self.assertTupleEqual(tensor.shape, expected_tensor.shape)
        self.assertTrue(torch.allclose(tensor, expected_tensor))

    def test_concatenated_on_3_sequences(self):
        tensor = self.few_seqs_batch.concatenated()
        expected_tensor = torch.tensor([
            [1], [4], [4], [4], [2], [4.]
        ])
        self.assertTupleEqual(tensor.shape, expected_tensor.shape)
        self.assertTrue(torch.allclose(tensor, expected_tensor))

    def test_concatenate_batch_must_be_of_the_same_shape_as_tensor_except_last_dimension(self):
        batch = torch.zeros(3, 2, 1)
        self.assertRaises(Exception, lambda: self.few_seqs_batch.concatenate_batch(batch))

        batch = torch.zeros(2, 3, 1)
        self.assertRaises(Exception, lambda: self.few_seqs_batch.concatenate_batch(batch))

        batch = torch.zeros(3, 3, 2)
        self.few_seqs_batch.concatenate_batch(batch)

    def test_concatenate_batch(self):
        batch = torch.tensor([
            [[1], [2], [3.]],
            [[4], [5], [6]],
            [[7], [8], [9]]
        ])

        expected_tensor = torch.tensor([
            [1], [4], [5], [6], [7], [8.]
        ])
        tensor = self.few_seqs_batch.concatenate_batch(batch)
        self.assertTupleEqual(expected_tensor.shape, tensor.shape)
        self.assertTrue(torch.allclose(expected_tensor, tensor))
