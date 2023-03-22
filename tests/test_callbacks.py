import os
import shutil
import unittest

import torch

import handwriting_synthesis.callbacks
from handwriting_synthesis.data import Tokenizer
from handwriting_synthesis.models import SynthesisNetwork
from handwriting_synthesis.sampling import HandwritingSynthesizer


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

        checkpoint = handwriting_synthesis.callbacks.EpochModelCheckpoint(model, self._save_dir, save_interval=1)

        checkpoint.on_iteration(epoch=0, epoch_iteration=10, iteration=25)
        self.assertFalse(os.path.exists(self._save_dir))

    def test_epoch_checkpoint_will_save_model_on_epoch_end_at_right_intervals(self):
        tokenizer = Tokenizer('abcd')
        alphabet_size = tokenizer.size
        device = torch.device('cpu')

        model = SynthesisNetwork.get_default_model(alphabet_size, device)
        model = model.to(device)

        mu = torch.zeros(2, dtype=torch.float32)
        sd = torch.ones(2, dtype=torch.float32)
        synthesizer = HandwritingSynthesizer(
            model, mu, sd, tokenizer.charset, num_steps=200
        )

        checkpoint = handwriting_synthesis.callbacks.EpochModelCheckpoint(
            synthesizer, self._save_dir, save_interval=3
        )
        checkpoint.on_epoch(0)
        save_path = os.path.join(self._save_dir, f'model_at_epoch_{0}.pt')
        self.assertFalse(os.path.exists(save_path))

        epoch = 2
        checkpoint.on_epoch(epoch)
        save_dir = os.path.join(self._save_dir, f'Epoch_{3}')
        self.assertTrue(os.path.isdir(save_dir))

        checkpoint.on_epoch(5)
        save_dir = os.path.join(self._save_dir, f'Epoch_{6}')
        self.assertTrue(os.path.isdir(save_dir))
