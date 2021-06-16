import argparse
import os
from handwriting_synthesis import data
from handwriting_synthesis.data_providers import IAMonDBProviderFactory


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("save_dir", type=str, help="Directory to save training and validation datasets")
    parser.add_argument("num_examples", type=int, help="Training set size")
    parser.add_argument("-l", "--max_len", type=int, default=50, help="Max number of points in a handwriting sequence")
    args = parser.parse_args()

    save_dir = args.save_dir
    num_examples = args.num_examples
    max_len = args.max_len

    train_save_path = os.path.join(save_dir, 'train.h5')
    val_save_path = os.path.join(save_dir, 'val.h5')
    os.makedirs(save_dir, exist_ok=True)

    factory = IAMonDBProviderFactory(num_examples)
    train_provider = factory.train_data_provider
    data.build_dataset(train_provider, train_save_path, max_len)
    print('Prepared training data')

    val_provider = factory.val_data_provider
    data.build_dataset(val_provider, val_save_path, max_len)
    print('Prepared validation data')
