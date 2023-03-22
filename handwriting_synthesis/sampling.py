import re
import json
import os
import torch
from .utils import visualize_strokes, plot_attention_weights
from .data import transcriptions_to_tensor, Tokenizer
from . import models


class UnconditionalSampler:
    @classmethod
    def load(cls, model_dir, device, bias):
        model_path = cls.get_model_path(model_dir)
        meta_path = cls.get_meta_path(model_dir)

        with open(meta_path, 'r') as f:
            s = f.read()

        d = json.loads(s)
        mu = torch.tensor(d['mu'], dtype=torch.float32)
        sd = torch.tensor(d['std'], dtype=torch.float32)
        charset = d['charset']

        tokenizer = Tokenizer(charset)

        alphabet_size = tokenizer.size

        model = cls.create_model_instance(alphabet_size, device, bias)
        model = model.to(device)

        # todo: verify this would work for GPU device as well

        if device.type == 'cpu':
            state_dict = torch.load(model_path, map_location=device)
        else:
            state_dict = torch.load(model_path)

        model.load_state_dict(state_dict)
        return cls(model, mu, sd, charset, num_steps=1500)

    @classmethod
    def create_model_instance(cls, alphabet_size, device, bias):
        return models.HandwritingPredictionNetwork.get_default_model(device, bias=bias)

    @classmethod
    def load_latest(cls, check_points_dir, device, bias=0):
        if not os.path.isdir(check_points_dir):
            print(f'Cannot load a model because directory {check_points_dir} does not exist')
            return None, 0

        most_recent = ''
        largest_epoch = 0
        for dir_name in os.listdir(check_points_dir):
            matches = re.findall(r'Epoch_([\d]+)', dir_name)
            if not matches:
                continue

            iteration_number = int(matches[0])
            if iteration_number > largest_epoch:
                largest_epoch = iteration_number
                most_recent = dir_name

        if most_recent:
            recent_checkpoint = os.path.join(check_points_dir, most_recent)
            sampler = cls.load(recent_checkpoint, device, bias)
            print(f'Loaded model weights from {recent_checkpoint} file')
            return sampler, largest_epoch
        else:
            print(f'Could not find a model')
            return None, 0

    @classmethod
    def get_model_path(cls, model_dir):
        return os.path.join(model_dir, 'model.pt')

    @classmethod
    def get_meta_path(cls, model_dir):
        return os.path.join(model_dir, 'meta.json')

    def __init__(self, model, mu, sd, charset, num_steps):
        self.model = model
        self.num_steps = num_steps
        self.mu = mu
        self.sd = sd
        self.tokenizer = Tokenizer(charset)

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)

        model_path = self.get_model_path(model_dir)
        meta_path = self.get_meta_path(model_dir)

        torch.save(self.model.state_dict(), model_path)
        means = self.mu.cpu().tolist()
        stds = self.sd.cpu().tolist()

        meta_data = {
            'mu': means,
            'std': stds,
            'charset': self.tokenizer.charset
        }

        s = json.dumps(meta_data)

        with open(meta_path, 'w') as f:
            f.write(s)

    def generate_handwriting(self, text='', output_path=None, thickness=10):
        output_path = output_path or self.derive_file_name(text)

        c = self._encode_text(text) if text else None

        sampled_handwriting = self.model.sample_means(context=c, steps=self.num_steps,
                                                      stochastic=True)
        sampled_handwriting = sampled_handwriting.cpu()
        sampled_handwriting = self._undo_normalization(sampled_handwriting)
        visualize_strokes(sampled_handwriting, output_path, lines=True, thickness=thickness)

    def derive_file_name(self, text):
        extension = '.png'
        return re.sub('[^0-9a-zA-Z]+', '_', text) + extension

    def _encode_text(self, text):
        transcription_batch = [text]
        return transcriptions_to_tensor(self.tokenizer, transcription_batch)

    def _undo_normalization(self, tensor):
        return tensor * self.sd + self.mu


class HandwritingSynthesizer(UnconditionalSampler):
    @classmethod
    def create_model_instance(cls, alphabet_size, device, bias):
        return models.SynthesisNetwork.get_default_model(alphabet_size, device, bias=bias)

    def visualize_attention(self, text, output_path=None, thickness=10):
        output_path = output_path or self.derive_file_name(text)
        sentinel = ' '
        text = text + sentinel
        c = self._encode_text(text)

        sampled_handwriting, phi = self.model.sample_means_with_attention(context=c, steps=self.num_steps,
                                                                          stochastic=True)
        sampled_handwriting = sampled_handwriting.cpu()
        sampled_handwriting = self._undo_normalization(sampled_handwriting)

        plot_attention_weights(phi, sampled_handwriting, output_path, text=text,
                               thickness=thickness)
