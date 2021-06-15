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

    args = parser.parse_args()

    device = torch.device("cpu")
    model = training.HandwritingSynthesisTask(device)._model
    text = args.text

    output_path = re.sub('[^0-9a-zA-Z]+', '_', text)
    output_path = f'{output_path}.png'

    model.load_state_dict(torch.load(args.path, map_location=torch.device('cpu')))

    c = training.transcriptions_to_tensor([args.text])

    with data.H5Dataset(f'{args.data_dir}/train.h5') as dataset:
        mu = torch.tensor(dataset.mu)
        sd = torch.tensor(dataset.std)

    try:
        sampled_handwriting = model.sample_means(steps=300, stochastic=True, context=c)
        sampled_handwriting = sampled_handwriting * sd + mu
        utils.visualize_strokes(sampled_handwriting, output_path, lines=True)
    except Exception:
        traceback.print_exc()
