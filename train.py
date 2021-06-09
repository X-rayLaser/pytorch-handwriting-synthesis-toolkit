import argparse
import torch
from handwriting_synthesis import training
from handwriting_synthesis import data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("data_path", type=str, help="Directory containing training and validation data h5 files")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="# of epochs to train")
    parser.add_argument("-i", "--interval", type=int, default=100, help="Iterations between sampling")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    epochs = args.epochs
    sampling_interval = args.interval

    with data.H5Dataset('datasets/train.h5') as dataset:
        mu = dataset.mu
        sd = dataset.std

    with data.NormalizedDataset('datasets/train.h5', mu, sd) as train_set, \
            data.NormalizedDataset('datasets/val.h5', mu, sd) as val_set:
        num_train_examples = len(train_set)
        num_val_examples = len(val_set)
        max_length = train_set.max_length

        print(f'Training handwriting prediction model: training set size {num_train_examples}, '
              f'validation set size {num_val_examples}, '
              f'batch size {batch_size}, '
              f'max sequence length {max_length}')

        train_task = training.HandwritingPredictionTrainingTask(device)
        loop = training.TrainingLoop(train_set, batch_size=batch_size, training_task=train_task)

        cb = training.HandwritingGenerationCallback(
            train_task._model, 'samples', max_length,
            train_set, iteration_interval=sampling_interval
        )

        loop.add_callback(cb)
        loop.start(epochs)
