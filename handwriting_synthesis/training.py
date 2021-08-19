import math
import torch.nn.functional
from torch.utils.data.dataloader import DataLoader
from .metrics import MovingAverage
from .tasks import TrainingTask
from . import utils
from .utils import collate, compute_validation_loss, compute_validation_metrics
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
error_handler = logging.FileHandler(filename='epochs_stats.log')
error_handler.setLevel(logging.INFO)
logger.addHandler(error_handler)


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

                nats_loss = ma_loss.value

                iteration_record = Formatter.format_iteration_entry(
                    epoch, i, num_batches, nats_loss, self._train_metrics
                )

                self._output_device.write(f'\r{iteration_record}', end='')
                iteration += 1

            val_loss_nats = compute_validation_loss(self._trainer, self._val_set, val_batch_size)
            compute_validation_metrics(self._trainer, self._val_set, val_batch_size, self._val_metrics)

            s = Formatter.format_epoch_info(
                epoch, ma_loss.value, val_loss_nats, self._train_metrics, self._val_metrics
            )

            self._run_epoch_callbacks(epoch)
            self._output_device.write(f'\r{s}', end='\n')
            logger.info(s)

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

    def _run_iteration_callbacks(self, epoch, epoch_iteration, iteration):
        with torch.no_grad():
            for cb in self._callbacks:
                cb.on_iteration(epoch, epoch_iteration, iteration)

    def _run_epoch_callbacks(self, epoch):
        with torch.no_grad():
            for cb in self._callbacks:
                cb.on_epoch(epoch)

    def set_output_device(self, device):
        self._output_device = device

    def add_callback(self, cb):
        self._callbacks.append(cb)


class Formatter:
    @classmethod
    def format_iteration_entry(cls, epoch, iteration, num_iterations, training_loss, training_metrics):
        metric_string = cls._format_metrics(training_metrics)

        return f'Epoch {epoch:4} {iteration + 1}/{num_iterations} batches. ' \
               f'Loss {training_loss:7.2f} nats.' \
               f'{metric_string}'

    @classmethod
    def format_epoch_info(cls, epoch, loss, val_loss, train_metrics, val_metrics):
        all_metrics = ''
        if train_metrics:
            all_metrics = f' {cls._format_metrics(train_metrics)}'

        if val_metrics:
            all_metrics = f'{all_metrics} {cls._format_metrics(val_metrics, val_metrics=True)}'

        return f'Epoch {epoch:4} Loss {loss:7.2f} nats, Val. loss {val_loss:7.2f} nats.' \
               f'{all_metrics}'

    @classmethod
    def _format_metrics(cls, metrics, val_metrics=False):
        prefix = 'Val. ' if val_metrics else ''

        formatted_values = [f'{prefix}{metric.name} {metric.value:6.4f}' for metric in metrics]
        s = '. '.join(formatted_values)
        if formatted_values:
            s = f' {s}.'
        return s


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
