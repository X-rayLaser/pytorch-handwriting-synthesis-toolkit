import argparse
import os
from handwriting_synthesis import data
from handwriting_synthesis.data_providers import IAMonDBProviderFactory


def calculate_max_length(train_size, val_size):
    factory = IAMonDBProviderFactory(train_size, val_size)

    print(f'Calculating maximum length of training set sequences.')
    train_max_len = data.get_max_sequence_length(factory.train_data_provider)

    print(f'Calculating maximum length of validation set sequences.')
    val_max_len = data.get_max_sequence_length(factory.val_data_provider)

    return max(train_max_len, val_max_len)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("save_dir", type=str, help="Directory to save training and validation datasets")
    parser.add_argument("train_size", type=int, help="Training set size")
    parser.add_argument("val_size", type=int, help="Validation set size")
    parser.add_argument(
        "-l", "--max_len", type=int, default=-1,
        help="Truncate sequences to be at most max_len long. No truncation is applied by default"
    )
    args = parser.parse_args()
    max_len = args.max_len
    save_dir = args.save_dir

    train_save_path = os.path.join(save_dir, 'train.h5')
    val_save_path = os.path.join(save_dir, 'val.h5')
    os.makedirs(save_dir, exist_ok=True)

    if max_len == -1:
        max_len = calculate_max_length(args.train_size, args.val_size)
        print(f'Maximum length is {max_len} points')

    factory = IAMonDBProviderFactory(args.train_size, args.val_size)
    data.build_dataset(factory.train_data_provider, train_save_path, max_len)
    print('Prepared training data')

    data.build_dataset(factory.val_data_provider, val_save_path, max_len)
    print('Prepared validation data')
