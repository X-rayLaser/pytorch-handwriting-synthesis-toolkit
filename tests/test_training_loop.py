import math
import unittest

from handwriting_synthesis import training


class TrainingLoopTests(unittest.TestCase):
    def test_loop_prints_epochs(self):
        ds = [(i, i**2) for i in range(3)]
        loop = training.TrainingLoop(ds, batch_size=1)

        output_device = training.InMemoryDevice()
        loop.set_output_device(output_device)
        loop.start(epochs=2)
        self.assertIn(f'Epoch {0:4} finished', output_device.lines[0])
        self.assertIn(f'Epoch {1:4} finished', output_device.lines[1])

    def test_loop_prints_epoch_progress(self):
        ds = [(i, i**2) for i in range(2)]
        loop = training.TrainingLoop(ds, batch_size=1)

        output_device = training.InMemoryDevice()
        loop.set_output_device(output_device)

        it = loop.get_iterator(epochs=1)

        next(it)
        self.assertIn(f'Epoch {0:4} 1/2 batches', output_device.lines[0])

        next(it)
        self.assertIn(f'Epoch {0:4} 2/2 batches', output_device.lines[0])

        next(it)
        self.assertIn(f'Epoch {0:4} finished', output_device.lines[0])

    def test_loop_prints_metrics_after_every_epoch(self):
        ds = [(i, i**2) for i in range(2)]
        loop = training.TrainingLoop(ds, batch_size=1)

        dummy_loss = 23.32
        task = training.DummyTask(dummy_loss)

        output_device = training.InMemoryDevice()
        loop.set_output_device(output_device)
        loop.set_training_task(task)
        loop.start(epochs=2)

        expected_loss = dummy_loss / math.log2(math.e)
        expected = [
            f"\rEpoch {0:4} finished. Loss {expected_loss:7.2f} nats.",
            f"\rEpoch {0:4} finished. Loss {expected_loss:7.2f} nats."
        ]
        self.assertIn(expected[0], output_device.lines[0])

    def test_callback_is_invoked(self):
        ds = [(0, 0), (1, 1)]
        loop = training.TrainingLoop(ds, batch_size=1)

        on_iteration_args = []
        on_epoch_args = []

        class Foo(training.Callback):
            def on_iteration(self, epoch, epoch_iteration, iteration):
                on_iteration_args.append((epoch, epoch_iteration, iteration))

            def on_epoch(self, epoch):
                on_epoch_args.append(epoch)

        loop.add_callback(Foo())
        loop.start(epochs=2)

        self.assertEqual([(0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)], on_iteration_args)
        self.assertEqual([0, 1], on_epoch_args)