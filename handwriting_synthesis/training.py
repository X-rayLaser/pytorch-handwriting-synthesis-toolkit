import math
import torch.nn.functional
from torch.utils.data.dataloader import DataLoader
from .metrics import MovingAverage
from .tasks import TrainingTask


def collate(batch):
    x = []
    y = []
    for points, text in batch:
        x.append(points)
        y.append(text)
    return x, y


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

    def start(self, initial_epoch, epochs):
        it = self.get_iterator(initial_epoch, epochs)
        for _ in it:
            pass

    def get_iterator(self, initial_epoch, epochs):
        loader = DataLoader(self._dataset, self._batch_size, collate_fn=collate)
        num_batches = math.ceil(len(self._dataset) / self._batch_size)

        iteration = 0

        ma_loss = MovingAverage()

        for epoch in range(initial_epoch, initial_epoch + epochs):
            ma_loss.reset()

            for i, data in enumerate(loader):
                y_hat, loss = self._trainer.train(data)
                self._run_iteration_callbacks(epoch, i, iteration)

                with torch.no_grad():
                    ma_loss.update(loss)

                nats_loss = ma_loss.nats
                self._output_device.write(
                    f'\rEpoch {epoch:4} {i + 1}/{num_batches} batches. Loss {nats_loss:7.2f} nats',
                    end=''
                )
                iteration += 1
                yield

            s = self._format_epoch_info(epoch, ma_loss.nats)

            self._run_epoch_callbacks(epoch)
            self._output_device.write(f'\r{s}', end='\n')
            yield

    def _run_iteration_callbacks(self, epoch, epoch_iteration, iteration):
        with torch.no_grad():
            for cb in self._callbacks:
                cb.on_iteration(epoch, epoch_iteration, iteration)

    def _run_epoch_callbacks(self, epoch):
        with torch.no_grad():
            for cb in self._callbacks:
                cb.on_epoch(epoch)

    def _format_epoch_info(self, epoch, loss):
        return f'Epoch {epoch:4} finished. Loss {loss:7.2f} nats.'

    def set_training_task(self, task):
        self._trainer = task

    def set_output_device(self, device):
        self._output_device = device

    def add_callback(self, cb):
        self._callbacks.append(cb)


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
