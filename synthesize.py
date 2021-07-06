import re
import os
import argparse
import torch

import handwriting_synthesis.data
from handwriting_synthesis import data, utils, models, callbacks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Path to prepared dataset directory")
    parser.add_argument("model_path", type=str, help="Path to saved model")
    parser.add_argument("text", type=str, help="Text to be converted to handwriting")
    parser.add_argument(
        "-b", "--bias",  type=float, default=0, help="A probability bias. Unbiased sampling is performed by default."
    )
    parser.add_argument("--trials",  type=int, default=1, help="Number of attempts")
    parser.add_argument(
        "--show_weights", default=False, action="store_true",
        help="When set, will produce a plot: handwriting against attention weights"
    )
    parser.add_argument("-c", "--charset", type=str, default='', help="Path to the charset file")
    parser.add_argument("--samples_dir", type=str, default='samples',
                        help="Path to the directory that will store samples")

    args = parser.parse_args()

    default_charset_path = os.path.join(args.data_dir, 'charset.txt')
    charset_path = utils.get_charset_path_or_raise(args.charset, default_charset_path)

    tokenizer = data.Tokenizer.from_file(charset_path)
    alphabet_size = tokenizer.size

    device = torch.device("cpu")

    bias = args.bias
    model = models.SynthesisNetwork.get_default_model(alphabet_size, device, bias=bias)
    model = model.to(device)

    base_file_name = re.sub('[^0-9a-zA-Z]+', '_', args.text)

    model.load_state_dict(torch.load(args.model_path, map_location=device))

    sentinel = '\n'
    full_text = args.text + sentinel
    c = handwriting_synthesis.data.transcriptions_to_tensor(tokenizer, [full_text])

    with data.H5Dataset(f'{args.data_dir}/train.h5') as dataset:
        mu = torch.tensor(dataset.mu)
        sd = torch.tensor(dataset.std)

    synthesizer = utils.HandwritingSynthesizer(
        model, mu, sd, num_steps=dataset.max_length, stochastic=True
    )

    output_dir = args.samples_dir
    os.makedirs(output_dir, exist_ok=True)

    for i in range(1, args.trials + 1):
        output_path = os.path.join(output_dir, f'{base_file_name}_{i}.png')
        synthesizer.synthesize(c, output_path, show_attention=args.show_weights, text=full_text)
        print(f'Done {i} / {args.trials}')
