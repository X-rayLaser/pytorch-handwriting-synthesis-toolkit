import argparse
import torch
from handwriting_synthesis import training
from handwriting_synthesis import data, utils


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
    _, largest_epoch = utils.load_saved_weights(model, check_points_dir=config.model_path)
    loop.add_callback(training.EpochModelCheckpoint(model, config.model_path, save_interval=1))

    loop.start(initial_epoch=largest_epoch, epochs=config.epochs)


def train_unconditional_handwriting_generator(train_set, val_set, config):
    train_task = training.HandwritingPredictionTrainingTask(device)
    train_task.load_model_weights(config.model_path)

    cb = training.HandwritingGenerationCallback(
        train_task._model, 'samples', max_length,
        train_set, iteration_interval=config.sampling_interval
    )

    train_model(train_set, val_set, train_task, [cb], config,
                training_task_verbose='Training (unconditional) handwriting prediction model')


def train_handwriting_synthesis_model(train_set, val_set, config):
    train_task = training.HandwritingSynthesisTask(device)
    train_task.load_model_weights(config.model_path)

    cb = training.HandwritingSynthesisCallback(
        10,
        train_task._model, 'synthesized', max_length,
        train_set, iteration_interval=config.sampling_interval
    )

    train_model(train_set, val_set, train_task, [cb], config,
                training_task_verbose='Training handwriting synthesis model')


def get_device():
    dev = torch.device("cpu")
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
    else:
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
            # computations on TPU are very slow for some reason
            dev = xm.xla_device()
        except ImportError:
            pass
    return dev


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("data_path", type=str, help="Directory containing training and validation data h5 files")
    parser.add_argument("model_path", type=str, help="Path for storing model weights")
    parser.add_argument(
        "-s", "--synthesis", default=False, action="store_true",
        help="Whether or not to train synthesis network (unconditional prediction network is trained by default)"
    )
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="# of epochs to train")
    parser.add_argument("-i", "--interval", type=int, default=100, help="Iterations between sampling")

    args = parser.parse_args()

    device = get_device()

    print(f'Using device {device}')

    with data.H5Dataset(f'{args.data_path}/train.h5') as dataset:
        mu = dataset.mu
        sd = dataset.std

    with data.NormalizedDataset(f'{args.data_path}/train.h5', mu, sd) as train_set, \
            data.NormalizedDataset(f'{args.data_path}/val.h5', mu, sd) as val_set:
        num_train_examples = len(train_set)
        num_val_examples = len(val_set)
        max_length = train_set.max_length
        model_path = args.model_path

        config = ConfigOptions(batch_size=args.batch_size, epochs=args.epochs,
                               sampling_interval=args.interval, num_train_examples=num_train_examples,
                               num_val_examples=num_val_examples, max_length=max_length,
                               model_path=model_path)

        if args.synthesis:
            train_handwriting_synthesis_model(train_set, val_set, config)
        else:
            train_unconditional_handwriting_generator(train_set, val_set, config)
