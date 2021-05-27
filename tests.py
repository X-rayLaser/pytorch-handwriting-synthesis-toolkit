import unittest
import training
import torch
import shutil
import os


class MovingAverageTests(unittest.TestCase):
    def test_after_initialization(self):
        ma = training.MovingAverage()
        self.assertEqual(ma.value, 0)

    def test_after_single_update(self):
        ma = training.MovingAverage()
        ma.update(25)
        self.assertEqual(ma.value, 25)

    def test_after_few_updates(self):
        ma = training.MovingAverage()
        ma.update(3)
        ma.update(5)
        ma.update(1)
        self.assertEqual(ma.value, 3)

    def test_reset_erases_previous_value(self):
        ma = training.MovingAverage()
        ma.update(3)
        ma.update(5)
        ma.reset()
        self.assertEqual(ma.value, 0)


class MetricTests(unittest.TestCase):
    def test_mse_name(self):
        self.assertEqual(training.MSE().name, 'MSE')

    def test_mse_is_zero_after_initialization(self):
        metric = training.MSE()
        self.assertEqual(metric.value, 0)

    def test_mse_on_a_single_example(self):
        metric = training.MSE()

        y_hat = torch.tensor([[2., 3.]])
        ground_true = torch.tensor([[5., 7.]])
        metric.update(y_hat, ground_true)
        # 12.5 = (5 - 2)**2 + (7 - 3) ** 2
        self.assertEqual(metric.value, 12.5)

    def test_mse_on_a_batch(self):
        metric = training.MSE()

        y_hat = torch.tensor([[2., 3.], [1., 1.]])
        ground_true = torch.tensor([[5., 7.], [2., 2.]])
        metric.update(y_hat, ground_true)
        self.assertEqual(27 / 4, metric.value)


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

        expected = [
            f"\rEpoch {0:4} finished. Loss   23.32.",
            f"\rEpoch {0:4} finished. Loss   23.32."
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
        self.fail('Finish it')


if __name__ == '__main__':
    unittest.main()


# todo: test save and load model weights
# todo: metrics
# todo: loss
# todo: visualization (PDF heatmap, window)
# todo: dataset
# todo: model
