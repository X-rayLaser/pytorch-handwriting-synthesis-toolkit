import os
import re
import torch
from handwriting_synthesis import utils
from handwriting_synthesis.data import transcriptions_to_tensor


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

        os.makedirs(self._save_dir, exist_ok=True)

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
        mu, std = self.train_set.mu, self.train_set.std
        mu = torch.tensor(mu)
        std = torch.tensor(std)
        synthesizer = utils.HandwritingSynthesizer(self.model, mu, std, num_steps=steps, stochastic=stochastic)
        synthesizer.synthesize(c=context, output_path=save_path, show_attention=False)


class HandwritingSynthesisCallback(HandwritingGenerationCallback):
    def __init__(self, tokenizer, images_per_iterations=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.images_per_iteration = images_per_iterations
        self.tokenizer = tokenizer

        self.mu = torch.tensor(self.train_set.mu)
        self.std = torch.tensor(self.train_set.std)

    def get_names_with_contexts(self, iteration):
        images_per_iteration = min(len(self.train_set), self.images_per_iteration)
        res = []
        for i in range(images_per_iteration):
            _, transcription = self.train_set[i]
            transcription_batch = [transcription]

            name = re.sub('[^0-9a-zA-Z]+', '_', transcription)
            file_name = f'{name}.png'
            context = transcriptions_to_tensor(self.tokenizer, transcription_batch)
            res.append((file_name, context))

        return res

    def generate_handwriting(self, save_path, steps, stochastic=True, context=None):
        super().generate_handwriting(save_path, steps, stochastic=stochastic, context=context)

        path, ext = os.path.splitext(save_path)
        save_path = f'{path}_attention{ext}'

        synthesizer = utils.HandwritingSynthesizer(
            self.model, self.mu, self.std, num_steps=steps, stochastic=stochastic
        )
        synthesizer.synthesize(c=context, output_path=save_path, show_attention=True)
