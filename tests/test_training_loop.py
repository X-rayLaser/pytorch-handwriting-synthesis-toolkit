import unittest

import handwriting_synthesis.callbacks
import handwriting_synthesis.tasks
from handwriting_synthesis import training


class TrainingLoopTests(unittest.TestCase):
    def test_loop_prints_epochs(self):
        ds = [(i, i**2) for i in range(3)]
        loop = training.TrainingLoop(ds, ds, batch_size=1)

        output_device = training.InMemoryDevice()
        loop.set_output_device(output_device)
        loop.start(0, epochs=2)
        self.assertIn(f'Epoch {0:4} finished', output_device.lines[0])
        self.assertIn(f'Epoch {1:4} finished', output_device.lines[1])

    def test_callback_is_invoked(self):
        ds = [(0, 0), (1, 1)]
        loop = training.TrainingLoop(ds, ds, batch_size=1)

        on_iteration_args = []
        on_epoch_args = []

        class Foo(handwriting_synthesis.callbacks.Callback):
            def on_iteration(self, epoch, epoch_iteration, iteration):
                on_iteration_args.append((epoch, epoch_iteration, iteration))

            def on_epoch(self, epoch):
                on_epoch_args.append(epoch)

        loop.add_callback(Foo())
        loop.start(0, epochs=2)

        self.assertEqual([(0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)], on_iteration_args)
        self.assertEqual([0, 1], on_epoch_args)