import os
import re
import json
import torch
from handwriting_synthesis import utils
from handwriting_synthesis.data import transcriptions_to_tensor


class Callback:
    def on_iteration(self, epoch, epoch_iteration, iteration):
        pass

    def on_epoch(self, epoch):
        pass


class EpochModelCheckpoint(Callback):
    def __init__(self, synthesizer, save_dir, save_interval):
        self._synthesizer = synthesizer
        self._save_dir = save_dir
        self._save_interval = save_interval

    def on_epoch(self, epoch):
        if (epoch + 1) % self._save_interval == 0:
            epoch_dir = os.path.join(self._save_dir, f'Epoch_{epoch + 1}')
            self._synthesizer.save(epoch_dir)


class HandwritingGenerationCallback(Callback):
    def __init__(self, model, samples_dir, max_length, dataset, iteration_interval=10):
        self.model = model
        self.samples_dir = samples_dir
        self.max_length = max_length
        self.interval = iteration_interval
        self.dataset = dataset

    def on_iteration(self, epoch, epoch_iteration, iteration):
        if (iteration + 1) % self.interval == 0:
            steps = self.max_length

            random_dir = os.path.join(self.samples_dir, 'random')
            os.makedirs(random_dir, exist_ok=True)

            names_with_contexts = self.get_names_with_contexts(iteration)

            if len(names_with_contexts) > 1:
                random_dir = os.path.join(random_dir, str(iteration))
                os.makedirs(random_dir, exist_ok=True)

            for file_name, context, text in names_with_contexts:
                random_path = os.path.join(random_dir, file_name)

                with torch.no_grad():
                    self.generate_handwriting(random_path, steps=steps, stochastic=True, context=context, text=text)

    def get_names_with_contexts(self, iteration):
        file_name = f'iteration_{iteration}.png'
        context = None
        text = ''
        return [(file_name, context, text)]

    def generate_handwriting(self, save_path, steps, stochastic=True, context=None, text=''):
        mu, std = self.dataset.mu, self.dataset.std
        mu = torch.tensor(mu)
        std = torch.tensor(std)
        synthesizer = utils.HandwritingSynthesizer(self.model, mu, std, num_steps=steps, stochastic=stochastic)
        synthesizer.synthesize(c=context, output_path=save_path, show_attention=False)


class HandwritingSynthesisCallback(HandwritingGenerationCallback):
    def __init__(self, tokenizer, images_per_iterations=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.images_per_iteration = images_per_iterations
        self.tokenizer = tokenizer

        self.mu = torch.tensor(self.dataset.mu)
        self.std = torch.tensor(self.dataset.std)

    def get_names_with_contexts(self, iteration):
        images_per_iteration = min(len(self.dataset), self.images_per_iteration)
        res = []
        sentinel = '\n'
        for i in range(images_per_iteration):
            _, transcription = self.dataset[i]

            text = transcription + sentinel

            transcription_batch = [text]

            name = re.sub('[^0-9a-zA-Z]+', '_', transcription)
            file_name = f'{name}.png'
            context = transcriptions_to_tensor(self.tokenizer, transcription_batch)
            res.append((file_name, context, text))

        return res

    def generate_handwriting(self, save_path, steps, stochastic=True, context=None, text=''):
        super().generate_handwriting(save_path, steps, stochastic=stochastic, context=context)

        path, ext = os.path.splitext(save_path)
        save_path = f'{path}_attention{ext}'

        synthesizer = utils.HandwritingSynthesizer(
            self.model, self.mu, self.std, num_steps=steps, stochastic=stochastic
        )
        synthesizer.synthesize(c=context, output_path=save_path, show_attention=True, text=text)
