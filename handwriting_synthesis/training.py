import math
import torch.nn.functional
from torch.utils.data.dataloader import DataLoader
from .metrics import MovingAverage
from .tasks import TrainingTask
from . import utils


def collate(batch):
    x = []
    y = []
    for points, text in batch:
        x.append(points)
        y.append(text)
    return x, y


def compute_validation_loss(trainer, dataset, batch_size):
    loader = DataLoader(dataset, batch_size, collate_fn=collate)

    with torch.no_grad():
        ma_loss = MovingAverage()
        ma_loss.reset()

        for i, data in enumerate(loader):
            y_hat, loss = trainer.compute_loss(data)
            ma_loss.update(loss)

        return ma_loss.nats


def compute_validation_metrics(trainer, dataset, batch_size, metrics):
    loader = DataLoader(dataset, batch_size, collate_fn=collate)

    with torch.no_grad():
        for metric in metrics:
            metric.reset()

        for i, data in enumerate(loader):
            y_hat, loss = trainer.compute_loss(data)
            _, eos = y_hat
            points, _ = data
            for metric in metrics:
                ground_true = utils.PaddedSequencesBatch(points, device=eos.device)
                metric.update(y_hat, ground_true)


class TrainingLoop:
    def __init__(self, dataset, validation_dataset, batch_size, training_task=None,
                 train_metrics=None, val_metrics=None):
        self._dataset = dataset
        self._val_set = validation_dataset
        self._output_device = ConsoleDevice()
        self._batch_size = batch_size

        if training_task is None:
            self._trainer = TrainingTask()
        else:
            self._trainer = training_task

        self._callbacks = []
        self._train_metrics = train_metrics or []
        self._val_metrics = val_metrics or []

    def start(self, initial_epoch, epochs):
        it = self.get_iterator(initial_epoch, epochs)
        for _ in it:
            pass

    def get_iterator(self, initial_epoch, epochs):
        loader = DataLoader(self._dataset, self._batch_size, collate_fn=collate)
        num_batches = math.ceil(len(self._dataset) / self._batch_size)
        val_batch_size = min(self._batch_size, len(self._val_set))

        iteration = 0

        ma_loss = MovingAverage()

        for epoch in range(initial_epoch, initial_epoch + epochs):
            ma_loss.reset()
            self._reset_metrics()

            for i, data in enumerate(loader):
                points, _ = data
                y_hat, loss = self._trainer.train(data)
                self._run_iteration_callbacks(epoch, i, iteration)

                with torch.no_grad():
                    ma_loss.update(loss)
                    self._compute_train_metrics(y_hat, points)

                nats_loss = ma_loss.nats
                train_metrics = self._format_metrics(self._train_metrics)
                self._output_device.write(
                    f'\rEpoch {epoch:4} {i + 1}/{num_batches} batches. Loss {nats_loss:7.2f} nats.{train_metrics}',
                    end=''
                )
                iteration += 1
                yield

            val_loss_nats = compute_validation_loss(self._trainer, self._val_set, val_batch_size)
            compute_validation_metrics(self._trainer, self._val_set, val_batch_size, self._val_metrics)

            s = self._format_epoch_info(epoch, ma_loss.nats, val_loss_nats)

            self._run_epoch_callbacks(epoch)
            self._output_device.write(f'\r{s}', end='\n')
            yield

    def _reset_metrics(self):
        for metric in self._train_metrics:
            metric.reset()

        for metric in self._val_metrics:
            metric.reset()

    def _compute_train_metrics(self, y_hat, points):
        mixture, eos = y_hat
        ground_true = utils.PaddedSequencesBatch(points, device=eos.device)

        for metric in self._train_metrics:
            metric.update(y_hat, ground_true)

    def _format_metrics(self, metrics, val_metrics=False):
        prefix = 'Val. ' if val_metrics else ''

        formatted_values = [f'{prefix}{metric.name} {metric.value:6.4f}' for metric in metrics]
        s = '. '.join(formatted_values)
        if formatted_values:
            s = f' {s}.'
        return s

    def _run_iteration_callbacks(self, epoch, epoch_iteration, iteration):
        with torch.no_grad():
            for cb in self._callbacks:
                cb.on_iteration(epoch, epoch_iteration, iteration)

    def _run_epoch_callbacks(self, epoch):
        with torch.no_grad():
            for cb in self._callbacks:
                cb.on_epoch(epoch)

    def _format_epoch_info(self, epoch, loss, val_loss):
        train_metrics = self._format_metrics(self._train_metrics)
        val_metrics = self._format_metrics(self._val_metrics, val_metrics=True)

        all_metrics = ''
        if self._train_metrics:
            all_metrics = f' {train_metrics}'

        if self._val_metrics:
            all_metrics = f'{all_metrics} {val_metrics}'

        return f'Epoch {epoch:4} Loss {loss:7.2f} nats, Val. loss {val_loss:7.2f}.' \
               f'{all_metrics}'

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


# todo: MSE metrics run on training and validation datasets
