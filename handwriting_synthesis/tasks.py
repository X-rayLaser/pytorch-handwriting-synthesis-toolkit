import torch
from handwriting_synthesis import losses, utils, models, data
from handwriting_synthesis.data import transcriptions_to_tensor
from handwriting_synthesis.optimizers import CustomRMSprop


class TrainingTask:
    def train(self, batch):
        return 0, 0

    def compute_loss(self, batch):
        return 0, 0


class DummyTask(TrainingTask):
    def __init__(self, hardcoded_loss=0.232):
        self._loss = hardcoded_loss

    def train(self, batch):
        return batch, self._loss


class BaseHandwritingTask(TrainingTask):
    def __init__(self, device, model, clip_values=None):
        self._device = device
        self._model = model
        self._model = self._model.to(self._device)
        self._optimizer = CustomRMSprop(
            self._model.parameters(), lr=0.0001, alpha=0.95, eps=10 ** (-4),
            momentum=9000, centered=True
        )

        self._clip_values = clip_values

    def train(self, batch):
        self._optimizer.zero_grad()
        y_hat, loss = self.compute_loss(batch)
        loss.backward()
        if self._clip_values:
            output_clip_value, lstm_clip_value = self._clip_values
            self._model.clip_gradients(
                output_clip_value=output_clip_value,
                lstm_clip_value=lstm_clip_value
            )
        self._optimizer.step()

        return y_hat, loss

    def compute_loss(self, batch):
        inputs, ground_true = self.prepare_batch(batch)

        y_hat = self._model(*inputs)
        mixtures, eos_hat = y_hat

        y_hat = (mixtures, eos_hat)
        loss = losses.nll_loss(mixtures, eos_hat, ground_true)
        return y_hat, loss

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
    pass


class HandwritingSynthesisTask(HandwritingPredictionTrainingTask):
    def __init__(self, tokenizer, *args, **kwargs):
        super(HandwritingSynthesisTask, self).__init__(*args, **kwargs)
        self._tokenizer = tokenizer

    def get_extra_input(self, transcriptions):
        tensor = transcriptions_to_tensor(self._tokenizer, transcriptions)
        tensor = tensor.to(device=self._device)
        return (tensor,)


# todo: redesign
