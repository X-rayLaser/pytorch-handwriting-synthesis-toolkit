import os
import argparse
import torch
from handwriting_synthesis import data, utils, models, tasks, metrics
from handwriting_synthesis.sampling import UnconditionalSampler, HandwritingSynthesizer


def evaluate_loss_and_metrics(task, dataset, batch_size=16):
    loss = utils.compute_validation_loss(task, dataset, batch_size=batch_size, verbose=True)
    all_metrics = [metrics.MSE(), metrics.SSE()]
    utils.compute_validation_metrics(task, dataset, batch_size=batch_size, metrics=all_metrics, verbose=True)
    return loss, all_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Computes a loss and other metrics of trained network on validation set.'
    )
    parser.add_argument("data_dir", type=str, help="Path to prepared dataset directory")
    parser.add_argument("path", type=str, help="Path to a saved model")
    parser.add_argument(
        "-u", "--unconditional", default=False, action="store_true",
        help="Whether or not the model is unconditional (assumes conditional model by default)"
    )

    args = parser.parse_args()

    device = torch.device("cpu")

    if args.unconditional:
        sampler = UnconditionalSampler.load(args.path, device, bias=0)
    else:
        sampler = HandwritingSynthesizer.load(args.path, device, bias=0)

    model = sampler.model
    mu = sampler.mu
    sd = sampler.sd
    tokenizer = sampler.tokenizer

    if args.unconditional:
        task = tasks.HandwritingPredictionTrainingTask(device, model)
    else:
        task = tasks.HandwritingSynthesisTask(tokenizer, device, model)

    train_dataset_path = os.path.join(args.data_dir, 'train.h5')
    val_dataset_path = os.path.join(args.data_dir, 'val.h5')

    with data.NormalizedDataset(val_dataset_path, mu, sd) as val_set:
        val_loss, val_metrics = evaluate_loss_and_metrics(task, val_set)

        print(f'Validation loss and metrics: Loss {val_loss}, '
              f'MSE {val_metrics[0].value}, SSE {val_metrics[1].value}')
