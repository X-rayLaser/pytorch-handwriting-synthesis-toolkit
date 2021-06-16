import torch
from handwriting_synthesis import losses, utils, models, data
from handwriting_synthesis.callbacks import transcriptions_to_tensor
from handwriting_synthesis.optimizers import CustomRMSprop


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
        self._model = self._model.to(self._device)
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

    def load_model_weights(self, path):
        model, epochs = utils.load_saved_weights(self._model, path)
        self._model = model.to(self._device)

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


class HandwritingSynthesisTask(HandwritingPredictionTrainingTask):
    def get_model(self, device):
        alphabet_size = data.Tokenizer().size
        return models.SynthesisNetwork(3, 400, alphabet_size, device)

    def get_extra_input(self, transcriptions):
        tensor = transcriptions_to_tensor(transcriptions)
        tensor = tensor.to(device=self._device)
        return (tensor,)
