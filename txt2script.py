import re
import os
import argparse
import torch

import handwriting_synthesis.data
from handwriting_synthesis import data, utils, models, callbacks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts a text file into a handwriting page.')
    parser.add_argument("data_dir", type=str, help="Path to prepared dataset directory")
    parser.add_argument("model_path", type=str, help="Path to saved model")
    parser.add_argument(
        "input_path", type=str, help="A path to a text file that needs to be converted to a handwriting")
    parser.add_argument(
        "-b", "--bias",  type=float, default=0, help="A probability bias. Unbiased sampling is performed by default."
    )

    parser.add_argument("-c", "--charset", type=str, default='', help="Path to the charset file")
    parser.add_argument("--output_path", type=str, default='',
                        help="Path to the generated handwriting file "
                             "(by default, it will be saved to the current working directory "
                             "whose name will be input_path with trailing .png extension)")
    args = parser.parse_args()

    if not os.path.isfile(args.input_path):
        raise Exception(f'Text file not found: {args.input_path}')

    default_charset_path = os.path.join(args.data_dir, 'charset.txt')
    charset_path = utils.get_charset_path_or_raise(args.charset, default_charset_path)

    tokenizer = data.Tokenizer.from_file(charset_path)
    alphabet_size = tokenizer.size

    device = torch.device("cpu")

    bias = args.bias
    model = models.SynthesisNetwork.get_default_model(alphabet_size, device, bias=bias)
    model = model.to(device)

    base_file_name = re.sub('[^0-9a-zA-Z]+', '_', args.input_path)
    output_path = args.output_path or f'{base_file_name}_.png'

    model.load_state_dict(torch.load(args.model_path, map_location=device))

    with data.H5Dataset(f'{args.data_dir}/train.h5') as dataset:
        mu = torch.tensor(dataset.mu)
        sd = torch.tensor(dataset.std)

    with open(args.input_path) as f:
        text = f.read()

    utils.text_to_document(model, mu, sd, tokenizer, text, output_path)
