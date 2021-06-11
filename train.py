import argparse
import torch
from handwriting_synthesis import training
from handwriting_synthesis import data


class ConfigOptions:
    def __init__(self, batch_size, epochs, sampling_interval,
                 num_train_examples, num_val_examples, max_length,
                 model_path):
        self.batch_size = batch_size
        self.epochs = epochs
        self.sampling_interval = sampling_interval
        self.num_train_examples = num_train_examples
        self.num_val_examples = num_val_examples
        self.max_length = max_length
        self.model_path = model_path


def print_info_message(training_task_verbose, config):
    print(f'{training_task_verbose} with options: training set size {config.num_train_examples}, '
          f'validation set size {config.num_val_examples}, '
          f'batch size {config.batch_size}, '
          f'max sequence length {config.max_length},'
          f'sampling interval (in # epochs): {config.sampling_interval}')


def train_model(train_set, val_set, train_task, callbacks, config, training_task_verbose):
    print_info_message(training_task_verbose, config)

    loop = training.TrainingLoop(train_set, batch_size=config.batch_size, training_task=train_task)

    for cb in callbacks:
        loop.add_callback(cb)

    model = loop._trainer._model
    loop.add_callback(training.EpochModelCheckpoint(model, config.model_path, save_interval=1))
    loop.start(config.epochs)


def train_unconditional_handwriting_generator(train_set, val_set, config):
    train_task = training.HandwritingPredictionTrainingTask(device)
    train_task.load_model_weights(config.model_path)

    cb = training.HandwritingGenerationCallback(
        train_task._model, 'samples', max_length,
        train_set, iteration_interval=config.sampling_interval
    )

    train_model(train_set, val_set, train_task, [cb], config,
                training_task_verbose='Training handwriting prediction model')


def train_handwriting_synthesis_model(train_set, val_set, config):
    train_task = training.HandwritingSynthesisTask(device)
    train_task.load_model_weights(config.model_path)

    cb = training.HandwritingSynthesisCallback(
        10,
        train_task._model, 'synthesized', max_length,
        train_set, iteration_interval=config.sampling_interval
    )

    train_model(train_set, val_set, train_task, [cb], config,
                training_task_verbose='Training handwriting prediction model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("data_path", type=str, help="Directory containing training and validation data h5 files")
    parser.add_argument("model_path", type=str, help="Path for storing model weights")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="# of epochs to train")
    parser.add_argument("-i", "--interval", type=int, default=100, help="Iterations between sampling")

    args = parser.parse_args()

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
        except ImportError:
            pass

    print(f'Will be running on device {device}')

    with data.H5Dataset('datasets/train.h5') as dataset:
        mu = dataset.mu
        sd = dataset.std

    with data.NormalizedDataset('datasets/train.h5', mu, sd) as train_set, \
            data.NormalizedDataset('datasets/val.h5', mu, sd) as val_set:
        num_train_examples = len(train_set)
        num_val_examples = len(val_set)
        max_length = train_set.max_length
        model_path = args.model_path

        config = ConfigOptions(args.batch_size, args.epochs, args.interval,
                               num_train_examples, num_val_examples, max_length,
                               model_path)

        unconditional = False
        if unconditional:
            train_unconditional_handwriting_generator(train_set, val_set, config)
        else:
            train_handwriting_synthesis_model(train_set, val_set, config)
