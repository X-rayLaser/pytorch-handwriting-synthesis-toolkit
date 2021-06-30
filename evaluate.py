import os
import argparse
import torch
from handwriting_synthesis import data, utils, models, tasks, metrics


def evaluate_loss_and_metrics(task, dataset, batch_size=16):
    loss = utils.compute_validation_loss(task, dataset, batch_size=batch_size, verbose=True)
    all_metrics = [metrics.MSE(), metrics.SSE()]
    utils.compute_validation_metrics(task, dataset, batch_size=batch_size, metrics=all_metrics, verbose=True)
    return loss, all_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Path to prepared dataset directory")
    parser.add_argument("path", type=str, help="Path to saved model")
    parser.add_argument("-c", "--charset", type=str, default='', help="Path to the charset file")
    parser.add_argument(
        "-u", "--unconditional", default=False, action="store_true",
        help="Whether or not the model is unconditional"
    )

    args = parser.parse_args()

    default_charset_path = os.path.join(args.data_dir, 'charset.txt')
    charset_path = utils.get_charset_path_or_raise(args.charset, default_charset_path)

    tokenizer = data.Tokenizer.from_file(charset_path)
    alphabet_size = tokenizer.size

    device = torch.device("cpu")

    if args.unconditional:
        model = models.HandwritingPredictionNetwork.get_default_model(device)
    else:
        model = models.SynthesisNetwork.get_default_model(alphabet_size, device)

    model = model.to(device)
    model.load_state_dict(torch.load(args.path, map_location=device))

    if args.unconditional:
        task = tasks.HandwritingPredictionTrainingTask(device, model)
    else:
        task = tasks.HandwritingSynthesisTask(tokenizer, device, model)

    with data.H5Dataset(f'{args.data_dir}/train.h5') as dataset:
        mu = dataset.mu
        sd = dataset.std

    train_dataset_path = os.path.join(args.data_dir, 'train.h5')
    val_dataset_path = os.path.join(args.data_dir, 'val.h5')

    with data.NormalizedDataset(val_dataset_path, mu, sd) as val_set:
        val_loss, val_metrics = evaluate_loss_and_metrics(task, val_set)

        print(f'Validation loss and metrics: Loss {val_loss}, '
              f'MSE {val_metrics[0].value}, SSE {val_metrics[1].value}')
