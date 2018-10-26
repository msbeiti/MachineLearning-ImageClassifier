# MachineLearning-ImageClassifier
An image classifier project using machine learning implemented via the pytorch python library. 

# Flower-Image-Classifier-Pytorch

This code is my submission for Udacity's AI Programming with Python Nanodegree. In this project, I created a command line application utilizing Pytorch, Neural Networks and Transfer Learning to train a new classifier on top of a pre-existing model trained on ImageNet to identify between 102 different types of flowers.

_This program can be modified to train itself on many different image classification problems if the proper arguments are passed.
Please read the comment comments to get information on arguments and how to use the programs._

## Example Commands
```
python train.py --arch densenet --hidden_units 1000,500 --epochs 10 --learning_rate 0.001 --gpu true --data_dir flowers/ --save_dir checkpoint.pth
```
```
python predict.py --checkpoint checkpoint.pth --input flowers/test/32/image_051001.jpg --top_k 5 --category_names cat_to_name.json --gpu true
```

## Training
To train a model, run `train.py` with the desired model architecture (densenet or vgg) and the path to the image folder:

```
python train.py --arch densenet --data_dir flowers [image folder with train, val and test sub-folders]
```
The command above will use default values for all other values. See below for how to customize these values.

### Usage
```
usage: train.py [-h] [--data_dir DATA_DIR] [--save_dir SAVE_DIR] [--arch ARCH]
                [--learning_rate LEARNING_RATE] [--hidden_units HIDDEN_UNITS]
                [--epochs EPOCHS] [--gpu GPU]

Provide image_dir, save_dir, architecture, hyperparameters such as
learningrate, num of hidden_units, epochs and whether to use gpu or not

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   path to image folder
  --save_dir SAVE_DIR   folder where model checkpoints gets saved to
  --arch ARCH           choose between vgg and densenet
  --learning_rate LEARNING_RATE
                        learning_rate for model
  --hidden_units HIDDEN_UNITS
                        hidden_units for model
  --epochs EPOCHS       epochs for model
  --gpu GPU             whether gpu should be used for or not
```

## Prediction
To make a prediction, run `predict.py` with the desired checkpoint and path to the image you want to try and predict:

```
python predict.py --checkpoint densenet201.pth --input flowers/test/23/image_05100.jpg
```
The command above will use default values for all other values. See below for how to customize these values.

### Usage
```
usage: predict.py [-h] [--input INPUT] [--checkpoint CHECKPOINT]
                  [--top_k TOP_K] [--category_names CATEGORY_NAMES]
                  [--gpu GPU]

Provide input, checkpoint, top_k, category_names and gpu

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         path to input image
  --checkpoint CHECKPOINT
                        path to checkpoint
  --top_k TOP_K         number of top_k to show
  --category_names CATEGORY_NAMES
                        path to cat names file
  --gpu GPU             whether gpu should be used for or not
```
