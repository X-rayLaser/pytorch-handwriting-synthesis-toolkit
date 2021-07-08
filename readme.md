![Handwriting image](results/Handwriting_synthesis_in_Python.png)

# Introduction

This toolkit contains utilities used to replicate some of the experiments described in Alex Graves's paper 
Generating Sequences With Recurrent Neural Networks. Concretely, this repository focuses on handwriting 
prediction and handwriting synthesis sections.

The repository includes almost everything that one might need for running the experiments. 
It provides a complete working pipeline to take the dataset, train the model and generate samples. 
One only needs to provide the dataset. Scripts will take of everything else, including data preprocessing and normalization.

The implementation closely follows the paper, from model architectures to training setup. 
However, it also provides some customization (such as choosing the batch size, 
the maximum length of the sequences, etc.)
In addition, the code can work with a custom dataset.

# Installation

After cloning this repository, install its python dependencies:
```
pip install -r requirements.txt
```

# Prerequisites

In order to carry out the experiments from the paper, you will need to download 
the IAM On-Line Handwriting Database (or shortly, IAM-OnDB).

It is also possible to work with a custom dataset, but one would require to implement a so-called data provider class
with just 2 methods. More on that is discussed in later sections.

You can obtain the IAM-OnDB dataset from here 
[The IAM On-Line Handwriting Database (IAM-OnDB)](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database).

Download the data set and unzip it into iam_ondb_home folder. The layout of the folder should be as follows:
```
├── ascii-all
│   └── ascii
├── lineImages-all
│   └── lineImages
├── lineStrokes-all
│   └── lineStrokes
├── original-xml-all
│   └── original
└── original-xml-part
    └── original
```

# Quickstart

Extract data examples from IAM-onDB dataset, preprocess it and save it into "data" directory:
```
python prepare_data.py data iam 9500 0 iam_ondb_home -l 700
```

After running this command, you should see a new folder called "data" containing 3 files:
```
.
├── charset.txt
├── train.h5
└── val.h5
```

Start training synthesis network for 50 epoch with a batch size 32 (this might a lot of time, even on GPU).
```
python train.py -b 32 -e 50 -i 300 data checkpoints
```

Create 1 handwriting for the string 'Text to be converted to handwriting'.
```
python synthesize.py data checkpoints/model_at_epoch_50.pt 'Text to be converted to handwriting ' -b 1
```

This section very briefly describes steps needed to train (conditional) synthesis network.
For more details, see dedicated sections below.

# Data preparation
The toolkit already comes with a built-in data preparation utility. 
However, it requires a so-called data provider. If you want to use the IAM-onDB dataset, 
no further action is necessary. Otherwise, you have to write your own to let the toolkit 
know how to extract the data.

Once there is a provider class, the toolkit will automatically preprocess data and save it. 
The preprocessing mainly involves the following steps:
- flattening every raw handwriting into a list of 3-element tuples  
(containing x and y coordinates as well as End-Of-Stroke flag)
- replacing every coordinate with its offset from the previous one
- truncating the sequences longer than a specified threshold

The steps above apply only to the handwriting portion of the data. 
Transcripts remain unchanged.

Data preparation is done by running the command prepare_data.py. 
Executing the script will create two files in HDF5 format, 
1 for training and 1 for validation examples. Along the way, the script also 
computes the mean and standard deviation and extracts all unique characters 
from transcripts.

The command expects at least two arguments: a path to a directory that will 
store prepared data and a name of a data provider. The name must match the 
data provider's name attributes (for example, iam for IAMonDBProvider class).
The data provider class might have __init__ method that takes arguments 
(as is the case with iam provider). If that's the case, you need to pass them 
in when calling the script.

An optional parameter --max_len sets the maximum length of handwriting. 
Any handwriting longer than max_len is going to be truncated to max_len points.

For more details on the command usage, see the Commands section.

# Implementing custom data provider

The data provider is a class with a class attribute "name" and two methods: 
get_training_data and get_validation_data.

Methods have to return an iterable or generator of pairs of examples in a 
certain format. Other than that, these methods can contain any logic whatsoever.

Every example needs to be a tuple of size 2. The second element is a 
corresponding transcript as a Python string. The first element of the 
tuple stores handwriting represented as a list of strokes. A stroke is yet 
another list of tuples (x, y), where x and y are coordinates recorded by a 
pen moving on the screen surface. Here is an example of handwriting consisting 
of 3 strokes:
```
[
    [(1, 2), (1, 3), (2, 5)],
    [(10, 3), (15, 4), (18, 8)],
    [(22, 10), (20, 5)]
]
```

Let's create a dummy data provider that returns only one training and 
validation example. First, open a python module 
handwriting_synthesis.data_providers.custom.py. 
Implement a new class named DummyProvider in that module. 
```
class DummyProvider(Provider):
    name = 'dummy'

    def get_training_data(self):
        handwriting = [
            [(1, 2), (1, 3), (2, 5)],
            [(10, 3), (15, 4), (18, 8)],
            [(22, 10), (20, 5)]
        ]

        transcript = 'Hi'
        yield handwriting, transcript

    def get_validation_data(self):
        handwriting = [
            [(1, 2), (1, 3), (2, 5)],
            [(10, 3), (15, 4), (18, 8)],
            [(22, 10), (20, 5)]
        ]

        transcript = 'Hi'
        yield handwriting, transcript

```

Here we yield the same data from both methods. If we wanted, we could 
fetch some data from a file or even download them via a network. It does 
not matter how the data gets retrieved. The only thing that matters is what 
the data provider returns or yields.

Note that name attribute we added to the class. We can now use it to tell 
the preparation script to use the data provider class with that name.
It's time to test the provider:
```
python prepare_data.py temp_data dummy
```
You should see a directory temp_data containing HDF5 files and one text file 
with the text "Hi".

# Training

After preparing the data, running "train.py" script will start or
resume training a network.
You can train either handwriting prediction network (which can be 
used for unconditional sampling), or a synthesis
network (text->handwriting network).

Network will be trained on GPU 
If CUDA device is available, otherwise CPU will be used.

After every epoch, network weights will be saved in a specified location.
In case of interruption of the script, the script will 
load the model weights and continue training.

Finally, after specified number iterations, the script will sample 
a few hand writings from a network and save them in a folder.

## Using custom character set
By default, the script will use character set stored under <prepared_data_folder>/charset.txt.
This file contains all characters that were found by scanning the text part of the dataset.
This might include digits, punctuation as well as other non-letter characters.
You might want to restrict the set to contain only white-space and letters. Just create a new text file and
populate it with characters that you want your synthesis network to be able to produce.

# Sampling from Handwriting synthesis network

Once you have a trained synthesis model, you can use it to convert a plain text into a handwritten one.
This can be done with *synthesize.py* script. The script will take a text and generate a specified number of
handwritings which all will be stored in a specified directory 
(by default, it will store them in "samples" directory). If a directory does not exist yet, it will be created.

The following arguments are required:
- a path to the prepared dataset folder
- a path to the model checkpoint
- a text to be converted to the handwriting.

For example, let's run the script to synthesize 5 samples of handwritten text for a text "A handwriting sample ".
Assuming that we prepared the data in a folder "data" and that we saved our model under
"checkpoints/model_at_epoch_60.pt", we can run it this way:
```
python synthesize.py data checkpoints/model_at_epoch_60.pt "A handwriting sample " --trials 5
```

You can also use this script to visualize the attention weights predicted by the model at evey predicted point.
This can be helpful for debugging reasons as it can tell whether your model learned to pay attention to the
relevant parts of text string when generating points. To do that, you need to pass a flag --show_weights:
```
python synthesize.py data checkpoints/model_at_epoch_60.pt "A handwriting sample " --trials 5 --show_weights 
```

*Important note. This script assumes that your network was trained using a default character set stored under
<your prepared dataset folder>/charset.txt. However, if it is not the case, you need to pass another optional
argument, --charset (or just -c). This will make sure that the script will correctly tokenize the text using the same
character mapping that was used during the training.* 

You might ask why the script needs a path to the dataset folder. The reason for that is that 
the dataset folder contains normalization
parameters (mean and standard deviation) used during the training to normalize examples.
We trained the synthesis network to predict next normalized offset of the point given the previous offsets.
To convert these normalized offsets into actual coordinates, we need to reverse the preprocessing and normalization
steps.

## Biased sampling
You can control the tradeoff between diversity and quality of produced
samples by passing optional parameter -b (or --bias).
Higher values result in a cleaner, nicer looking hand writings,
while lower values result in less readable but more diverse samples.

By default, bias equals to 0 which corresponds to unbiased sampling.

# Sampling from a prediction network (unconditional)
To sample from a prediction network, you can use sample.py script.
It expects 3 required arguments: path to prepared dataset, path to pretrained
prediction network and path to directory that will store generated samples.

Run the command below to generate 1 handwriting and save it to samples_dir folder.
```
python sample.py data checkpoints/model_at_epoch_2.pt samples_dir
```

# Commands

## prepare_data.py
Extract (optionally split), preprocess and save data in specified destination folder.

```
$ python prepare_data.py --help
usage: prepare_data.py [-h] [-l MAX_LEN]
                       save_dir provider_name
                       [provider_args [provider_args ...]]

positional arguments:
  save_dir              Directory to save training and validation datasets
  provider_name         A short name used to lookup the corresponding factory
                        class
  provider_args         Variable number of arguments expected by a provider
                        __init__ method

optional arguments:
  -h, --help            show this help message and exit
  -l MAX_LEN, --max_len MAX_LEN
                        Truncate sequences to be at most max_len long. No
                        truncation is applied by default
```

## train.py

Start/resume training prediction or synthesis network.

```
$ python train.py --help
usage: train.py [-h] [-u] [-b BATCH_SIZE] [-e EPOCHS] [-i INTERVAL]
                [-c CHARSET] [--samples_dir SAMPLES_DIR] [--clip1 CLIP1]
                [--clip2 CLIP2]
                data_dir model_dir

positional arguments:
  data_dir              Directory containing training and validation data h5
                        files
  model_dir             Directory storing model weights

optional arguments:
  -h, --help            show this help message and exit
  -u, --unconditional   Whether or not to train synthesis network (synthesis
                        network is trained by default)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size
  -e EPOCHS, --epochs EPOCHS
                        # of epochs to train
  -i INTERVAL, --interval INTERVAL
                        Iterations between sampling
  -c CHARSET, --charset CHARSET
                        Path to the charset file
  --samples_dir SAMPLES_DIR
                        Path to the directory that will store samples
  --clip1 CLIP1         Gradient clipping value for output layer. When omitted
                        or set to zero, no clipping is done.
  --clip2 CLIP2         Gradient clipping value for lstm layers. When omitted
                        or set to zero, no clipping is done.
```

## evaluate.py
Compute loss and other metrics of trained network on validation set.
```
$ python evaluate.py --help
usage: evaluate.py [-h] [-c CHARSET] [-u] data_dir path

positional arguments:
  data_dir              Path to prepared dataset directory
  path                  Path to saved model

optional arguments:
  -h, --help            show this help message and exit
  -c CHARSET, --charset CHARSET
                        Path to the charset file
  -u, --unconditional   Whether or not the model is unconditional
```

## sample.py
Generate (unconditionally) samples from pretrained prediction network.
```
$ python sample.py --help
usage: sample.py [-h] [-b BIAS] [-s STEPS] [-t TRIALS]
                 data_dir path sample_dir

positional arguments:
  data_dir              Path to prepared dataset directory
  path                  Path to saved model
  sample_dir            Path to directory that will contain generated samples

optional arguments:
  -h, --help            show this help message and exit
  -b BIAS, --bias BIAS  A probability bias. Unbiased sampling is performed by
                        default.
  -s STEPS, --steps STEPS
                        Number of points in generated sequence
  -t TRIALS, --trials TRIALS
                        Number of attempts
```

## synthesize.py
Generate a handwriting(s) for a given string of text.
```
$ python synthesize.py --help
usage: synthesize.py [-h] [-b BIAS] [--trials TRIALS] [--show_weights]
                     [-c CHARSET] [--samples_dir SAMPLES_DIR]
                     data_dir model_path text

positional arguments:
  data_dir              Path to prepared dataset directory
  model_path            Path to saved model
  text                  Text to be converted to handwriting

optional arguments:
  -h, --help            show this help message and exit
  -b BIAS, --bias BIAS  A probability bias. Unbiased sampling is performed by
                        default.
  --trials TRIALS       Number of attempts
  --show_weights        When set, will produce a plot: handwriting against
                        attention weights
  -c CHARSET, --charset CHARSET
                        Path to the charset file
  --samples_dir SAMPLES_DIR
                        Path to the directory that will store samples
```


# Samples

## Biased samples

### Probability bias 2:
![image](results/conditional/biased/2/Biased_handwriting_sample__1.png)
![image](results/conditional/biased/2/Biased_handwriting_sample__2.png)
![image](results/conditional/biased/2/Biased_handwriting_sample__3.png)
![image](results/conditional/biased/2/Biased_handwriting_sample__4.png)
![image](results/conditional/biased/2/Biased_handwriting_sample__5.png)

### Probability bias 1:
![image](results/conditional/biased/1/Biased_handwriting_sample__1.png)
![image](results/conditional/biased/1/Biased_handwriting_sample__2.png)
![image](results/conditional/biased/1/Biased_handwriting_sample__3.png)
![image](results/conditional/biased/1/Biased_handwriting_sample__4.png)
![image](results/conditional/biased/1/Biased_handwriting_sample__5.png)

### Probability bias 0.5:
![image](results/conditional/biased/0.5/Biased_handwriting_sample__1.png)
![image](results/conditional/biased/0.5/Biased_handwriting_sample__2.png)
![image](results/conditional/biased/0.5/Biased_handwriting_sample__3.png)
![image](results/conditional/biased/0.5/Biased_handwriting_sample__4.png)
![image](results/conditional/biased/0.5/Biased_handwriting_sample__5.png)

### Probability bias 0.1:
![image](results/conditional/biased/0.1/Biased_handwriting_sample__1.png)
![image](results/conditional/biased/0.1/Biased_handwriting_sample__2.png)
![image](results/conditional/biased/0.1/Biased_handwriting_sample__3.png)
![image](results/conditional/biased/0.1/Biased_handwriting_sample__4.png)
![image](results/conditional/biased/0.1/Biased_handwriting_sample__5.png)

## Unbiased samples

![image](results/conditional/unbiased/Unbiased_handwriting_sample__1.png)
![image](results/conditional/unbiased/Unbiased_handwriting_sample__2.png)
![image](results/conditional/unbiased/Unbiased_handwriting_sample__3.png)
![image](results/conditional/unbiased/Unbiased_handwriting_sample__4.png)
![image](results/conditional/unbiased/Unbiased_handwriting_sample__5.png)
![image](results/conditional/unbiased/Unbiased_handwriting_sample__6.png)
![image](results/conditional/unbiased/Unbiased_handwriting_sample__7.png)
![image](results/conditional/unbiased/Unbiased_handwriting_sample__8.png)
![image](results/conditional/unbiased/Unbiased_handwriting_sample__9.png)
![image](results/conditional/unbiased/Unbiased_handwriting_sample__10.png)

# References

[1] [Alex Graves. Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850)

[2] [Liwicki, M. and Bunke, H.: IAM-OnDB - an On-Line English Sentence Database Acquired from Handwritten Text on a Whiteboard](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database#LiBu05-03)

# Support
If you find this repository useful, consider starring it by clicking at the ★ button.
It would be much appreciated.