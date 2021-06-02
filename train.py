import argparse
import torch
from handwriting_synthesis import training
from handwriting_synthesis import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("examples", type=int, help="Training set size")
    parser.add_argument("-l", "--max_len", type=int, default=50)
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="# of epochs to train")
    parser.add_argument("-i", "--interval", type=int, default=100, help="Iterations between sampling")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_length = args.max_len
    batch_size = args.batch_size
    num_examples = args.examples
    epochs = args.epochs
    sampling_interval = args.interval

    print(f'Training handwriting prediction model: dataset size {num_examples}, '
          f'batch size {batch_size}, '
          f'max sequence length {max_length}')

    dataset_factory = utils.DataSetFactory('../iam_ondb_home', num_examples=num_examples, max_length=max_length)
    train_set, val_set = dataset_factory.split_data()
    train_task = training.HandwritingPredictionTrainingTask(device)
    loop = training.TrainingLoop(train_set, batch_size=batch_size, training_task=train_task)

    cb = training.HandwritingGenerationCallback(
        train_task._model, 'samples', max_length,
        train_set, iteration_interval=sampling_interval
    )

    loop.add_callback(cb)
    loop.start(epochs)
