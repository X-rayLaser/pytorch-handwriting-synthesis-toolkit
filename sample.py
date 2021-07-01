import os
import argparse
import torch
from handwriting_synthesis import data, utils, models


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Path to prepared dataset directory")
    parser.add_argument("path", type=str, help="Path to saved model")
    parser.add_argument("sample_dir", type=str, help="Path to directory that will contain generated samples")
    parser.add_argument(
        "-b", "--bias",  type=float, default=0, help="A probability bias. Unbiased sampling is performed by default."
    )
    parser.add_argument("-s", "--steps", type=int, default=700, help="Number of points in generated sequence")
    parser.add_argument("-t", "--trials",  type=int, default=1, help="Number of attempts")

    args = parser.parse_args()

    device = torch.device("cpu")

    model = models.HandwritingPredictionNetwork.get_default_model(device, bias=args.bias)
    model = model.to(device)
    model.load_state_dict(torch.load(args.path, map_location=device))

    with data.H5Dataset(f'{args.data_dir}/train.h5') as dataset:
        mu = torch.tensor(dataset.mu)
        sd = torch.tensor(dataset.std)

    synthesizer = utils.HandwritingSynthesizer(
        model, mu, sd, num_steps=args.steps, stochastic=True
    )

    os.makedirs(args.sample_dir, exist_ok=True)

    for i in range(1, args.trials + 1):
        output_path = os.path.join(args.sample_dir, f'{i}.png')
        synthesizer.synthesize(c=None, output_path=output_path, show_attention=False)
        print(f'Done {i} / {args.trials}')
