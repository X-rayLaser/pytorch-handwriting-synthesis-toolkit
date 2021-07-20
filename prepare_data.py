import argparse
import os
from handwriting_synthesis import data
from handwriting_synthesis.data_providers import registry


def calculate_max_length(provider_class, *args):
    provider = provider_class(*args)
    return data.get_max_sequence_length(provider.get_training_data())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extracts (optionally splits), preprocesses and saves data in specified destination folder.'
    )
    parser.add_argument("save_dir", type=str, help="Directory to save training and validation datasets")
    parser.add_argument("provider_name", type=str, help="A short name used to lookup the corresponding factory class")
    parser.add_argument(
        'provider_args', nargs='*', help="Variable number of arguments expected by a provider __init__ method"
    )

    parser.add_argument(
        "-l", "--max_len", type=int, default=0,
        help="Truncate sequences to be at most max_len long. No truncation is applied by default"
    )

    args = parser.parse_args()

    max_len = args.max_len
    save_dir = args.save_dir

    train_save_path = os.path.join(save_dir, 'train.h5')
    val_save_path = os.path.join(save_dir, 'val.h5')
    os.makedirs(save_dir, exist_ok=True)

    provider_name = args.provider_name

    if provider_name not in registry:
        error_msg = f'Name {provider_name} does match a name of any registered providers: {registry}.\n' \
                    f'If you are trying to use a custom provider class, make sure you followed these steps:\n' \
                    f'1. The class has to be defined in handwriting_synthesis.data_providers.custom module.\n' \
                    f'2. The class has to inherit from either Factory or DataSplittingFactory classes.\n' \
                    f'3. The class needs to set the "name" attribute to a non-empty string.'
        raise Exception(error_msg)

    provider_class = registry[provider_name]

    if max_len == 0:
        print(f'Calculating maximum length of training set sequences.')
        max_len = calculate_max_length(provider_class, *args.provider_args)
        print(f'Maximum length is {max_len} points')

    provider = provider_class(*args.provider_args)
    data.build_dataset(provider.get_training_data(), train_save_path, max_len)
    print('Prepared training data')

    data.build_dataset(provider.get_validation_data(), val_save_path, max_len)
    print('Prepared validation data')

    charset_path = os.path.join(save_dir, 'charset.txt')
    data.build_and_save_charset(train_save_path, charset_path)
    print(f'Charset is saved to {charset_path}')
