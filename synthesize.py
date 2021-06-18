import re
import os
import argparse
import torch

import handwriting_synthesis.callbacks
import handwriting_synthesis.tasks
from handwriting_synthesis import data, utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Path to prepared dataset directory")
    parser.add_argument("path", type=str, help="Path to saved model")
    parser.add_argument("text", type=str, help="Text to be converted to handwriting")
    parser.add_argument("--trials",  type=int, default=1, help="Number of attempts")
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
    stochastic = not args.deterministic

    device = torch.device("cpu")
    model = handwriting_synthesis.tasks.HandwritingSynthesisTask(device)._model

    base_file_name = re.sub('[^0-9a-zA-Z]+', '_', args.text)

    model.load_state_dict(torch.load(args.path, map_location=device))

    c = handwriting_synthesis.callbacks.transcriptions_to_tensor([args.text])

    with data.H5Dataset(f'{args.data_dir}/train.h5') as dataset:
        mu = torch.tensor(dataset.mu)
        sd = torch.tensor(dataset.std)

    synthesizer = utils.HandwritingSynthesizer(model, mu, sd, num_steps=700, stochastic=stochastic)

    output_dir = 'samples'
    os.makedirs(output_dir, exist_ok=True)

    for i in range(args.trials):
        output_path = os.path.join(output_dir, f'{base_file_name}_{i}.png')
        synthesizer.synthesize(c, output_path, show_attention=args.show_weights)
        print(f'Done {i} / {args.trials}')

# todo: reimplement tokenizer (based alphabet file and fit on data)
