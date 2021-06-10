import math
import re
import os
import traceback
import torch.nn.functional
from torch.utils.data.dataloader import DataLoader
from . import utils, losses, models, data
from .optimizers import CustomRMSprop
from .utils import visualize_strokes
from .metrics import MovingAverage


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

    def start(self, epochs):
        it = self.get_iterator(epochs)
        for _ in it:
            pass

    def get_iterator(self, epochs):
        loader = DataLoader(self._dataset, self._batch_size, collate_fn=collate)
        num_batches = math.ceil(len(self._dataset) / self._batch_size)

        iteration = 0

        ma_loss = MovingAverage()

        for epoch in range(epochs):
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


class TrainingTask:
    def train(self, batch):
        return 0, 0


class DummyTask(TrainingTask):
    def __init__(self, hardcoded_loss=0.232):
        self._loss = hardcoded_loss

    def train(self, batch):
        return batch, self._loss


class BaseHandwritingTask(TrainingTask):
    def __init__(self, device):
        self._device = device
        self._model = self.get_model(device)
        self._model.to(self._device)
        #self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)
        self._optimizer = CustomRMSprop(
            self._model.parameters(), lr=0.0001, alpha=0.95, eps=10 ** (-4),
            momentum=9000, centered=True
        )

    def train(self, batch):
        # todo: write metrics code

        self._optimizer.zero_grad()
        inputs, ground_true = self.prepare_batch(batch)

        y_hat = self._model(*inputs)
        mixtures, eos_hat = y_hat

        y_hat = (mixtures, eos_hat)
        loss = losses.nll_loss(mixtures, eos_hat, ground_true)

        loss.backward()
        #model.clip_gradient()
        self._optimizer.step()

        return y_hat, loss

    def get_model(self, device):
        raise NotImplementedError

    def prepare_batch(self, batch):
        points, transcriptions = batch
        ground_true = utils.PaddedSequencesBatch(points, device=self._device)

        batch_size, steps, input_dim = ground_true.tensor.shape

        prefix = torch.zeros(batch_size, 1, input_dim, device=self._device)
        x = torch.cat([prefix, ground_true.tensor[:, :-1]], dim=1)
        inputs = (x,) + self.get_extra_input(transcriptions)

        return inputs, ground_true

    def get_extra_input(self, transcriptions):
        return tuple()


class HandwritingPredictionTrainingTask(BaseHandwritingTask):
    def get_model(self, device):
        return models.HandwritingPredictionNetwork(3, 900, 20, device)


def transcriptions_to_tensor(transcriptions):
    tokenizer = data.Tokenizer()

    eye = torch.eye(tokenizer.size)

    token_sequences = []
    for s in transcriptions:
        tokens = tokenizer.tokenize(s)
        token_sequences.append(eye[tokens].numpy().tolist())

    batch = utils.PaddedSequencesBatch(token_sequences)
    return batch.tensor


class HandwritingSynthesisTask(HandwritingPredictionTrainingTask):
    def get_model(self, device):
        alphabet_size = data.Tokenizer().size
        return models.SynthesisNetwork(3, 400, alphabet_size, device)

    def get_extra_input(self, transcriptions):
        tensor = transcriptions_to_tensor(transcriptions)
        tensor = tensor.to(device=self._device)
        return (tensor,)


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


class HandwritingGenerationCallback(Callback):
    def __init__(self, model, samples_dir, max_length, train_set, iteration_interval=10):
        self.model = model
        self.samples_dir = samples_dir
        self.max_length = max_length
        self.interval = iteration_interval
        self.train_set = train_set

    def on_iteration(self, epoch, epoch_iteration, iteration):
        if (iteration + 1) % self.interval == 0:
            steps = self.max_length

            greedy_dir = os.path.join(self.samples_dir, 'greedy')
            random_dir = os.path.join(self.samples_dir, 'random')

            os.makedirs(greedy_dir, exist_ok=True)
            os.makedirs(random_dir, exist_ok=True)

            names_with_contexts = self.get_names_with_contexts(iteration)

            if len(names_with_contexts) > 1:
                greedy_dir = os.path.join(greedy_dir, str(iteration))
                random_dir = os.path.join(random_dir, str(iteration))
                os.makedirs(greedy_dir, exist_ok=True)
                os.makedirs(random_dir, exist_ok=True)

            for file_name, context in names_with_contexts:
                greedy_path = os.path.join(greedy_dir, file_name)
                random_path = os.path.join(random_dir, file_name)

                with torch.no_grad():
                    self.generate_handwriting(greedy_path, steps=steps, stochastic=False, context=context)
                    self.generate_handwriting(random_path, steps=steps, stochastic=True, context=context)

    def get_names_with_contexts(self, iteration):
        file_name = f'iteration_{iteration}.png'
        context = None
        return [(file_name, context)]

    def generate_handwriting(self, save_path, steps, stochastic=True, context=None):
        try:
            sampled_handwriting = self.model.sample_means(steps=steps, stochastic=stochastic, context=context)
            sampled_handwriting = sampled_handwriting.cpu()
            sampled_handwriting = self.train_set.denormalize(sampled_handwriting)
            visualize_strokes(sampled_handwriting, save_path, lines=True)
        except Exception:
            traceback.print_exc()


class HandwritingSynthesisCallback(HandwritingGenerationCallback):
    def __init__(self, images_per_iterations=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.images_per_iteration = images_per_iterations

    def get_names_with_contexts(self, iteration):
        images_per_iteration = min(len(self.train_set), self.images_per_iteration)
        res = []
        for i in range(images_per_iteration):
            _, transcription = self.train_set[i]
            transcription_batch = [transcription]

            name = re.sub('[^0-9a-zA-Z]+', '_', transcription)
            file_name = f'{name}.png'
            context = transcriptions_to_tensor(transcription_batch)
            res.append((file_name, context))

        return res
