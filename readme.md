![Handwriting image](generated_samples/Handwriting_synthesis_in_Python.png)

# Introduction

Implementation of Alex Graves et al. paper, sections 4 (handwriting prediction) and 5 (handwriting synthesis).
This repository contains utilities to train neural networks and experiment with them.

# 1. Installation

After cloning this repository, install its python dependencies:
```
pip install -r requirements.txt
```

# 2. Prerequisites

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

# 3. Quickstart

Extract data examples from IAM-onDB dataset, preprocess it and save it into "data" directory:
```
python prepare_data.py data iam 2000 200 iam_ondb_home
```

After running this command, you should see a new folder called "data" containing 3 files:
```
.
├── charset.txt
├── train.h5
└── val.h5
```

Start training synthesis network for 10 epoch with batch size 32.
```
python train.py -b 32 -e 10 -i 300 data checkpoints
```

Create 1 handwriting for the string 'Text to be converted to handwriting'.
```
python synthesize.py data checkpoints/model_at_epoch_2.pt 'Text to be converted to handwriting ' -b 1
```

This section very briefly describes steps needed to train (conditional) synthesis network.
For more details, see dedicated sections below.

# 4. The guide

## Raw data format
The toolkit expects data as tuples (sequence, transcript). Here,
sequence is a list of strokes, and a transcript is a 
corresponding text. A stroke is (another) list of
tuples (x, y), where x and y are point coordinates.

## Preprocessing

Once there is a class that implements a thin API to access raw data,
the toolkit can use this class to automatically preprocess data and save it.
This is done by running a script "prepare_data.py". Running
the script will create 2 .h5 files for training and validation examples.
Additionally, this will also create a text file storing all the 
characters extracted from transcripts charset.txt.

## Training

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

## Sampling from Handwriting synthesis network

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

### Biased sampling
You can control the tradeoff between diversity and quality of produced
samples by passing optional parameter -b (or --bias).
Higher values result in a cleaner, nicer looking hand writings,
while lower values result in less readable but more diverse samples.

By default, bias equals to 0 which corresponds to unbiased sampling.

## 1.5 Sampling from a prediction network (unconditional)
To sample from a prediction network, you can use sample.py script.
It expects 3 required arguments: path to prepared dataset, path to pretrained
prediction network and path to directory that will store generated samples.

Run the command below to generate 1 handwriting and save it to samples_dir folder.
```
python sample.py data checkpoints/model_at_epoch_2.pt samples_dir
```

# 5. Commands manual

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
                [-c CHARSET] [--clip1 CLIP1] [--clip2 CLIP2]
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
                     [-c CHARSET]
                     data_dir path text

positional arguments:
  data_dir              Path to prepared dataset directory
  path                  Path to saved model
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
```


# 6. Samples

## Unbiased samples

![image](generated_samples/conditional/unbiased/Unbiased_handwriting_sample__1.png)
![image](generated_samples/conditional/unbiased/Unbiased_handwriting_sample__2.png)
![image](generated_samples/conditional/unbiased/Unbiased_handwriting_sample__3.png)
![image](generated_samples/conditional/unbiased/Unbiased_handwriting_sample__4.png)
![image](generated_samples/conditional/unbiased/Unbiased_handwriting_sample__5.png)
![image](generated_samples/conditional/unbiased/Unbiased_handwriting_sample__6.png)
![image](generated_samples/conditional/unbiased/Unbiased_handwriting_sample__7.png)
![image](generated_samples/conditional/unbiased/Unbiased_handwriting_sample__8.png)
![image](generated_samples/conditional/unbiased/Unbiased_handwriting_sample__9.png)
![image](generated_samples/conditional/unbiased/Unbiased_handwriting_sample__10.png)

## Slightly biased samples (bias = 0.1)

# References
