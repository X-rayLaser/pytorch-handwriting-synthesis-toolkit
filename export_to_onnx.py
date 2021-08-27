import torch
from torch.autograd import Variable
import json
import onnx_models as models
from handwriting_synthesis.sampling import HandwritingSynthesizer
from handwriting_synthesis.data import Tokenizer


device = torch.device('cpu')
model_dir = 'checkpoints/Epoch_46'
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

onnx_model_path = 'synthesis_network.onnx'

torch.onnx.export(model, (x, c, w, k, h1, c1, h2, c2, h3, c3, bias), onnx_model_path, verbose=True, opset_version=9,
                  input_names=['x', 'c', 'w', 'k', 'h1', 'c1', 'h2', 'c2', 'h3', 'c3', 'bias'],
                  output_names=['pi', 'mu', 'sd', 'ro', 'eos', 'w', 'k', 'h1', 'c1', 'h2', 'c2', 'h3', 'c3'],
                  dynamic_axes={'c': {1: 'sequence'}})
