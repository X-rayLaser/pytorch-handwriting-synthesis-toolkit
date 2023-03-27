import re
import os
import argparse
import torch

import handwriting_synthesis.data
from handwriting_synthesis import data, utils, models, callbacks
from handwriting_synthesis.sampling import HandwritingSynthesizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Converts a single line of text into a handwriting with a randomly chosen style'
    )

    parser.add_argument("model_path", type=str, help="Path to saved model")
    parser.add_argument("text", type=str, help="Text to be converted to handwriting")

    parser.add_argument(
        "-b", "--bias",  type=float, default=0, help="A probability bias. Unbiased sampling is performed by default."
    )
    parser.add_argument(
        "--trials",  type=int, default=1,
        help="Number of attempts"
    )
    parser.add_argument(
        "--show_weights", default=False, action="store_true",
        help="When set, will produce a plot: handwriting against attention weights"
    )
    parser.add_argument(
        "--heatmap", default=False, action="store_true",
        help="When set, will produce a heatmap for mixture density outputs"
    )

    parser.add_argument(
        "--samples_dir", type=str, default='samples',
        help="Path to the directory that will store samples"
    )

    parser.add_argument(
        "--thickness", type=int, default=10,
        help="Handwriting thickness in pixels. It is set to 10 by default."
    )

    parser.add_argument(
        "--output_file_type", type=str, default="png",
        help="file type for results, currently only png and svg are supported."
    )

    args = parser.parse_args()

    device = torch.device("cpu")

    synthesizer = HandwritingSynthesizer.load(args.model_path, device, args.bias)
    output_dir = args.samples_dir
    thickness = args.thickness
    print(args)
    output_file_type = args.output_file_type
    os.makedirs(output_dir, exist_ok=True)

    base_file_name = re.sub('[^0-9a-zA-Z]+', '_', args.text)

    if args.heatmap:
        output_path = os.path.join(output_dir, f'{base_file_name}_.png')
        sentinel = '\n'
        full_text = args.text + sentinel
        c = handwriting_synthesis.data.transcriptions_to_tensor(synthesizer.tokenizer, [full_text])
        utils.plot_mixture_densities(synthesizer.model, synthesizer.mu, synthesizer.sd, output_path, c)
    else:
        for i in range(1, args.trials + 1):
            output_path = os.path.join(output_dir, f'{base_file_name}_{i}.{output_file_type}')
            if args.show_weights:
                synthesizer.visualize_attention(args.text, output_path, thickness=thickness)
            else:
                synthesizer.generate_handwriting(args.text, output_path, thickness=thickness)
            print(f'Done {i} / {args.trials}')
