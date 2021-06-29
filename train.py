import argparse
import os

import torch

import handwriting_synthesis.callbacks
import handwriting_synthesis.tasks
from handwriting_synthesis import training
from handwriting_synthesis import data, utils, models, metrics


class ConfigOptions:
    def __init__(self, batch_size, epochs, sampling_interval,
                 num_train_examples, num_val_examples, max_length,
                 model_path, charset_path, output_clip_value, lstm_clip_value):
        self.batch_size = batch_size
        self.epochs = epochs
        self.sampling_interval = sampling_interval
        self.num_train_examples = num_train_examples
        self.num_val_examples = num_val_examples
        self.max_length = max_length
        self.model_path = model_path
        self.charset_path = charset_path
        self.output_clip_value = output_clip_value
        self.lstm_clip_value = lstm_clip_value


def print_info_message(training_task_verbose, config):
    print(f'{training_task_verbose} with options: training set size {config.num_train_examples}, '
          f'validation set size {config.num_val_examples}, '
          f'batch size {config.batch_size}, '
          f'max sequence length {config.max_length},'
          f'sampling interval (in # epochs): {config.sampling_interval}')


def train_model(train_set, val_set, train_task, callbacks, config, training_task_verbose):
    print_info_message(training_task_verbose, config)

    train_metrics = [metrics.MSE(), metrics.SSE()]
    val_metrics = [metrics.MSE(), metrics.SSE()]

    loop = training.TrainingLoop(train_set, val_set, batch_size=config.batch_size, training_task=train_task,
                                 train_metrics=train_metrics, val_metrics=val_metrics)

    for cb in callbacks:
        loop.add_callback(cb)

    model = loop._trainer._model
    _, largest_epoch = utils.load_saved_weights(model, check_points_dir=config.model_path)
    loop.add_callback(handwriting_synthesis.callbacks.EpochModelCheckpoint(model, config.model_path, save_interval=1))

    loop.start(initial_epoch=largest_epoch, epochs=config.epochs)


def train_unconditional_handwriting_generator(train_set, val_set, device, config):
    model = models.HandwritingPredictionNetwork.get_default_model(device)
    model, epochs = utils.load_saved_weights(model, config.model_path)
    model = model.to(device)

    if config.output_clip_value == 0 or config.lstm_clip_value == 0:
        clip_values = None
    else:
        clip_values = (config.output_clip_value, config.lstm_clip_value)

    train_task = handwriting_synthesis.tasks.HandwritingPredictionTrainingTask(device, model, clip_values)

    cb = handwriting_synthesis.callbacks.HandwritingGenerationCallback(
        model, 'samples', config.max_length,
        train_set, iteration_interval=config.sampling_interval
    )

    train_model(train_set, val_set, train_task, [cb], config,
                training_task_verbose='Training (unconditional) handwriting prediction model')


def train_handwriting_synthesis_model(train_set, val_set, device, config):
    tokenizer = data.Tokenizer.from_file(config.charset_path)

    alphabet_size = tokenizer.size
    model = models.SynthesisNetwork.get_default_model(alphabet_size, device)
    model, epochs = utils.load_saved_weights(model, config.model_path)

    if config.output_clip_value == 0 or config.lstm_clip_value == 0:
        clip_values = None
    else:
        clip_values = (config.output_clip_value, config.lstm_clip_value)

    train_task = handwriting_synthesis.tasks.HandwritingSynthesisTask(
        tokenizer, device, model, clip_values
    )

    cb = handwriting_synthesis.callbacks.HandwritingSynthesisCallback(
        tokenizer,
        10,
        model, 'synthesized', config.max_length,
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

    parser.add_argument("data_dir", type=str, help="Directory containing training and validation data h5 files")
    parser.add_argument("model_dir", type=str, help="Directory storing model weights")
    parser.add_argument(
        "-u", "--unconditional", default=False, action="store_true",
        help="Whether or not to train synthesis network (synthesis network is trained by default)"
    )
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="# of epochs to train")
    parser.add_argument("-i", "--interval", type=int, default=100, help="Iterations between sampling")

    parser.add_argument(
        "--clip1", type=int, default=0,
        help="Gradient clipping value for output layer. "
             "When omitted or set to zero, no clipping is done."
    )
    parser.add_argument(
        "--clip2", type=int, default=0,
        help="Gradient clipping value for lstm layers. "
             "When omitted or set to zero, no clipping is done."
    )

    args = parser.parse_args()

    device = get_device()

    print(f'Using device {device}')

    with data.H5Dataset(f'{args.data_dir}/train.h5') as dataset:
        mu = dataset.mu
        sd = dataset.std

    train_dataset_path = os.path.join(args.data_dir, 'train.h5')
    val_dataset_path = os.path.join(args.data_dir, 'val.h5')
    charset_path = os.path.join(args.data_dir, 'charset.txt')

    with data.NormalizedDataset(train_dataset_path, mu, sd) as train_set, \
            data.NormalizedDataset(val_dataset_path, mu, sd) as val_set:
        num_train_examples = len(train_set)
        num_val_examples = len(val_set)
        max_length = train_set.max_length
        model_path = args.model_dir

        config = ConfigOptions(batch_size=args.batch_size, epochs=args.epochs,
                               sampling_interval=args.interval, num_train_examples=num_train_examples,
                               num_val_examples=num_val_examples, max_length=max_length,
                               model_path=model_path,
                               charset_path=charset_path,
                               output_clip_value=args.clip1, lstm_clip_value=args.clip2)

        if args.unconditional:
            train_unconditional_handwriting_generator(train_set, val_set, device, config)
        else:
            train_handwriting_synthesis_model(train_set, val_set, device, config)
