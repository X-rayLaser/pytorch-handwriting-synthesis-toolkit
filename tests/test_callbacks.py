import os
import shutil
import unittest

import torch

from handwriting_synthesis import training


class CallbackTests(unittest.TestCase):
    def setUp(self):
        self._save_dir = 'test_model_checkpoints'
        if os.path.isdir(self._save_dir):
            shutil.rmtree(self._save_dir)

    def tearDown(self):
        if os.path.isdir(self._save_dir):
            shutil.rmtree(self._save_dir)

    def test_callback_will_not_save_on_iteration(self):
        model = torch.nn.Sequential(torch.nn.Linear(1, 1))

        checkpoint = training.EpochModelCheckpoint(model, self._save_dir, save_interval=1)

        checkpoint.on_iteration(epoch=0, epoch_iteration=10, iteration=25)
        self.assertEqual(len(list(os.listdir(self._save_dir))), 0)

    def test_callback_will_not_save_on_epoch(self):
        model = torch.nn.Sequential(torch.nn.Linear(1, 1))

        checkpoint = training.IterationModelCheckpoint(model, self._save_dir, save_interval=1)

        checkpoint.on_epoch(epoch=4)
        self.assertEqual(len(list(os.listdir(self._save_dir))), 0)

    def test_epoch_checkpoint_will_save_model_on_epoch_end_at_right_intervals(self):
        model = torch.nn.Sequential(torch.nn.Linear(1, 1))
        checkpoint = training.EpochModelCheckpoint(model, self._save_dir, save_interval=3)
        checkpoint.on_epoch(0)
        save_path = os.path.join(self._save_dir, f'model_at_epoch_{0}.pt')
        self.assertFalse(os.path.exists(save_path))

        epoch = 2
        checkpoint.on_epoch(epoch)
        save_path = os.path.join(self._save_dir, f'model_at_epoch_{3}.pt')
        self.assertTrue(os.path.isfile(save_path))

        checkpoint.on_epoch(5)
        save_path = os.path.join(self._save_dir, f'model_at_epoch_{6}.pt')
        self.assertTrue(os.path.isfile(save_path))

    def test_iteration_checkpoint_will_save_model_periodically_after_every_few_iterations(self):
        #self.fail('Finish it')
        pass