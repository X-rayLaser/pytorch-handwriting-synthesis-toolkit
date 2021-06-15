import re
import argparse
import traceback
import torch
from handwriting_synthesis import data, utils, training


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Path to prepared dataset directory")
    parser.add_argument("path", type=str, help="Path to saved model")
    parser.add_argument("text", type=str, help="Text to be converted to handwriting")
    parser.add_argument(
        "--show_weights", default=False, action="store_true",
        help="When set, will produce a plot: handwriting against attention weights"
    )
    parser.add_argument(
        "--deterministic", default=False, action="store_true",
        help="When set, at every step will output a point with highest probability density. "
             "Otherwise, every point is randomly sampled"
    )

    args = parser.parse_args()
    text = args.text
    stochastic = not args.deterministic
    data_dir = args.data_dir
    model_path = args.path
    show_weights = args.show_weights

    device = torch.device("cpu")
    model = training.HandwritingSynthesisTask(device)._model

    output_path = re.sub('[^0-9a-zA-Z]+', '_', text)
    output_path = f'{output_path}.png'

    model.load_state_dict(torch.load(model_path, map_location=device))

    c = training.transcriptions_to_tensor([text])

    with data.H5Dataset(f'{data_dir}/train.h5') as dataset:
        mu = torch.tensor(dataset.mu)
        sd = torch.tensor(dataset.std)

    synthesizer = utils.HandwritingSynthesizer(model, mu, sd, num_steps=700, stochastic=stochastic)
    synthesizer.synthesize(c, output_path, show_attention=show_weights)
