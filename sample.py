import os
import argparse
import torch
from handwriting_synthesis.sampling import UnconditionalSampler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generates (unconditionally) samples from a pretrained prediction network.'
    )
    parser.add_argument("path", type=str, help="Path to saved model")
    parser.add_argument("sample_dir", type=str, help="Path to directory that will contain generated samples")
    parser.add_argument(
        "-b", "--bias",  type=float, default=0, help="A probability bias. Unbiased sampling is performed by default."
    )
    parser.add_argument("-s", "--steps", type=int, default=700, help="Number of points in generated sequence")
    parser.add_argument("-t", "--trials",  type=int, default=1, help="Number of attempts")

    parser.add_argument(
        "--thickness", type=int, default=10,
        help="Handwriting thickness in pixels. It is set to 10 by default."
    )

    args = parser.parse_args()

    device = torch.device("cpu")

    os.makedirs(args.sample_dir, exist_ok=True)

    sampler = UnconditionalSampler.load(args.path, device, args.bias)

    for i in range(1, args.trials + 1):
        output_path = os.path.join(args.sample_dir, f'{i}.png')
        sampler.generate_handwriting(output_path=output_path, thickness=args.thickness)
        print(f'Done {i} / {args.trials}')
