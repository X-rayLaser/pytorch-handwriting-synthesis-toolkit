import math
import os
import torch.nn.functional
from torch.utils.data.dataloader import DataLoader
import utils
import losses
import models


class TrainingLoop:
    def __init__(self, dataset, batch_size, training_task=None):
        self._dataset = dataset
        self._output_device = ConsoleDevice()
        self._batch_size = batch_size

        if training_task is None:
            self._trainer = TrainingTask()
        else:
            self._trainer = training_task
        self._callbacks = []

    def start(self, epochs):
        it = self.get_iterator(epochs)
        for _ in it:
            pass

    def get_iterator(self, epochs):
        loader = DataLoader(self._dataset, self._batch_size)
        num_batches = math.ceil(len(self._dataset) / self._batch_size)

        iteration = 0
        for epoch in range(epochs):
            for i, data in enumerate(loader):
                y_hat, loss = self._trainer.train(data)
                self._run_iteration_callbacks(epoch, i, iteration)
                self._output_device.write(f'\rEpoch {epoch:4} {i + 1}/{num_batches} batches.', end='')
                iteration += 1
                yield

            s = self._format_epoch_info(epoch, loss)

            self._run_epoch_callbacks(epoch)
            self._output_device.write(f'\r{s}', end='\n')
            yield

    def _run_iteration_callbacks(self, epoch, epoch_iteration, iteration):
        for cb in self._callbacks:
            cb.on_iteration(epoch, epoch_iteration, iteration)

    def _run_epoch_callbacks(self, epoch):
        for cb in self._callbacks:
            cb.on_epoch(epoch)

    def _format_epoch_info(self, epoch, loss):
        return f'Epoch {epoch:4} finished. Loss {loss:7.2f}.'

    def set_training_task(self, task):
        self._trainer = task

    def set_output_device(self, device):
        self._output_device = device

    def add_callback(self, cb):
        self._callbacks.append(cb)


class TrainingTask:
    def train(self, batch):
        return 0, 0


class DummyTask(TrainingTask):
    def __init__(self, hardcoded_loss=0.232):
        self._loss = hardcoded_loss

    def train(self, batch):
        return batch, self._loss


class HandwritingPredictionTrainingTask(TrainingTask):
    def __init__(self):
        self._model = models.HandwritingPredictionNetwork(3, 900, 20)

    def train(self, batch):
        points, transcriptions = batch
        ground_true = utils.PaddedSequencesBatch(points)

        batch_size, steps, input_dim = ground_true.tensor.shape

        prefix = torch.zeros(batch_size, 1, input_dim)
        x = torch.cat([prefix, ground_true.tensor[:, :-1]], dim=1)

        state = self._model.get_initial_state(batch_size)
        y_hat = self._model(x, state)
        mixtures, eos_hat = y_hat

        y_hat = (mixtures, eos_hat)
        loss = losses.nll_loss(mixtures, eos_hat, ground_true)

        loss.backward()
        loss.step()
        return y_hat, loss


class OutputDevice:
    def write(self, s, end='\n', **kwargs):
        pass


class ConsoleDevice(OutputDevice):
    def write(self, s, end='\n', **kwargs):
        print(s, end=end)


class InMemoryDevice(OutputDevice):
    def __init__(self):
        self._lines = ['']

    def write(self, s, end='\n', **kwargs):
        if s[0] == '\r':
            self._lines[-1] = s
        else:
            self._lines[-1] += s

        if end == '\n':
            self._lines.append('')

    @property
    def lines(self):
        return [line for line in self._lines if line]


class Callback:
    def on_iteration(self, epoch, epoch_iteration, iteration):
        pass

    def on_epoch(self, epoch):
        pass


class EpochModelCheckpoint(Callback):
    def __init__(self, model, save_dir, save_interval):
        self._model = model
        self._save_dir = save_dir
        self._save_interval = save_interval

        os.makedirs(self._save_dir, exist_ok=False)

    def on_epoch(self, epoch):
        if (epoch + 1) % self._save_interval == 0:
            save_path = os.path.join(self._save_dir, f'model_at_epoch_{epoch + 1}.pt')
            torch.save(self._model.state_dict(), save_path)


class IterationModelCheckpoint(Callback):
    def __init__(self, model, save_dir, save_interval):
        self._model = model
        self._save_dir = save_dir
        self._save_interval = save_interval

        os.makedirs(self._save_dir, exist_ok=False)

    def on_iteration(self, epoch, epoch_iteration, iteration):
        if (iteration + 1) % self._save_interval == 0:
            save_path = os.path.join(self._save_dir, f'model_at_epoch_{iteration + 1}.pt')
            torch.save(self._model.state_dict(), save_path)
