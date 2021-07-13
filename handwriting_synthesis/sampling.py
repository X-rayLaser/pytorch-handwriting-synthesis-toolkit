import re
import json
import os
import torch
from .utils import visualize_strokes, plot_attention_weights
from .data import transcriptions_to_tensor, Tokenizer
from . import models


class HandwritingSynthesizer:
    @classmethod
    def load(cls, model_dir, bias):
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

        device = torch.device("cpu")

        model = models.SynthesisNetwork.get_default_model(alphabet_size, device, bias=bias)
        model = model.to(device)

        model.load_state_dict(torch.load(model_path, map_location=device))

        return cls(model, mu, sd, charset, num_steps=1500)

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

    def generate_handwriting(self, text, output_path=None):
        output_path = output_path or self.derive_file_name(text)
        c = self._encode_text(text)

        sampled_handwriting = self.model.sample_means(context=c, steps=self.num_steps,
                                                      stochastic=True)
        sampled_handwriting = sampled_handwriting.cpu()
        sampled_handwriting = self._undo_normalization(sampled_handwriting)
        visualize_strokes(sampled_handwriting, output_path, lines=True)

    def visualize_attention(self, text, output_path=None):
        output_path = output_path or self.derive_file_name(text)
        sentinel = ' '
        text = text + sentinel
        c = self._encode_text(text)

        sampled_handwriting, phi = self.model.sample_means_with_attention(context=c, steps=self.num_steps,
                                                                          stochastic=True)
        sampled_handwriting = sampled_handwriting.cpu()
        sampled_handwriting = self._undo_normalization(sampled_handwriting)

        plot_attention_weights(phi, sampled_handwriting, output_path, text=text)

    def derive_file_name(self, text):
        extension = '.png'
        return re.sub('[^0-9a-zA-Z]+', '_', text) + extension

    def _encode_text(self, text):
        transcription_batch = [text]
        return transcriptions_to_tensor(self.tokenizer, transcription_batch)

    def _undo_normalization(self, tensor):
        return tensor * self.sd + self.mu
