import argparse
import os
import torch
from torch.autograd import Variable
import json
import onnx_models as models
from handwriting_synthesis.sampling import HandwritingSynthesizer
from handwriting_synthesis.data import Tokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Exports a given synthesis model to ONNX'
    )

    parser.add_argument("model_path", type=str, help="Path to a directory containing synthesis model")
    parser.add_argument("output_path", type=str, help="Output path for a resulting ONNX model")

    args = parser.parse_args()

    onnx_model_path = args.output_path
    output_dir_path, _ = os.path.split(onnx_model_path)
    os.makedirs(output_dir_path, exist_ok=True)

    device = torch.device('cpu')
    model_dir = args.model_path
    model_path = HandwritingSynthesizer.get_model_path(model_dir)
    meta_path = HandwritingSynthesizer.get_meta_path(model_dir)

    with open(meta_path, 'r') as f:
        s = f.read()

    d = json.loads(s)
    mu = torch.tensor(d['mu'], dtype=torch.float32)
    sd = torch.tensor(d['std'], dtype=torch.float32)
    charset = d['charset']

    tokenizer = Tokenizer(charset)

    alphabet_size = tokenizer.size

    model = models.SynthesisNetwork.get_default_model(alphabet_size, device, bias=0)
    model = model.to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Export the trained model to ONNX
    x = Variable(torch.randn(1, 1, 3, dtype=torch.float32))
    c = Variable(torch.randn(1, 1, alphabet_size, dtype=torch.float32))
    w = Variable(torch.randn(1, 1, alphabet_size, dtype=torch.float32))
    k = Variable(torch.randn(1, 10, dtype=torch.float32))
    h1 = Variable(torch.randn(1, 400, dtype=torch.float32))
    c1 = Variable(torch.randn(1, 400, dtype=torch.float32))
    h2 = Variable(torch.randn(1, 400, dtype=torch.float32))
    c2 = Variable(torch.randn(1, 400, dtype=torch.float32))
    h3 = Variable(torch.randn(1, 400, dtype=torch.float32))
    c3 = Variable(torch.randn(1, 400, dtype=torch.float32))
    bias = Variable(torch.randn(1, dtype=torch.float32))

    torch.onnx.export(model, (x, c, w, k, h1, c1, h2, c2, h3, c3, bias), onnx_model_path, verbose=True, opset_version=9,
                      input_names=['x', 'c', 'w', 'k', 'h1', 'c1', 'h2', 'c2', 'h3', 'c3', 'bias'],
                      output_names=['pi', 'mu', 'sd', 'ro', 'eos', 'w', 'k', 'h1', 'c1', 'h2', 'c2', 'h3', 'c3', 'phi'],
                      dynamic_axes={'c': {1: 'sequence'}, 'phi': {2: 'string_length'}})
