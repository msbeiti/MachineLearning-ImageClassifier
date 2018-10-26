
# Developing an AI application

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 

<img src='assets/Flowers.png' width=500px>

The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content

We'll lead you through each part which you'll implement in Python.

When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.

First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

Please make sure if you are running this notebook in the workspace that you have chosen GPU rather than CPU mode.


```python
# Imports here
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import json
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from collections import OrderedDict
from PIL import Image
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
```

## Load the data

Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.

The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.

The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
 


```python
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
```


```python
# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = {
    'train_transforms' : transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]),

    'valid_test_transforms' : transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    
}

# TODO: Load the datasets with ImageFolder
image_datasets = {'train_data': datasets.ImageFolder(train_dir, transform=data_transforms['train_transforms']),
                 'valid_data': datasets.ImageFolder(valid_dir, transform=data_transforms['valid_test_transforms']),
                  'test_data': datasets.ImageFolder(test_dir, transform=data_transforms['valid_test_transforms'])}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {'trainloader' : torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=64, shuffle=True),
               'validloader' : torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=32),
                'testloader' : torch.utils.data.DataLoader(image_datasets['test_data'], batch_size=32)}
```

### Label mapping

You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.


```python
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
```

# Building and training the classifier

Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.

We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:

* Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
* Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
* Train the classifier layers using backpropagation using the pre-trained network to get the features
* Track the loss and accuracy on the validation set to determine the best hyperparameters

We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!

When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.

One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.


```python
# TODO: Build and train your network
# Load the model
model = models.densenet161(pretrained=True)
model.type = 'densenet161'
model
```

    /opt/conda/lib/python3.6/site-packages/torchvision-0.2.1-py3.6.egg/torchvision/models/densenet.py:212: UserWarning: nn.init.kaiming_normal is now deprecated in favor of nn.init.kaiming_normal_.
    Downloading: "https://download.pytorch.org/models/densenet161-8d451a50.pth" to /root/.torch/models/densenet161-8d451a50.pth
    100%|██████████| 115730790/115730790 [00:08<00:00, 13297883.15it/s]





    DenseNet(
      (features): Sequential(
        (conv0): Conv2d(3, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        (norm0): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu0): ReLU(inplace)
        (pool0): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (denseblock1): _DenseBlock(
          (denselayer1): _DenseLayer(
            (norm1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(96, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer2): _DenseLayer(
            (norm1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(144, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer3): _DenseLayer(
            (norm1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer4): _DenseLayer(
            (norm1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(240, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer5): _DenseLayer(
            (norm1): BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(288, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer6): _DenseLayer(
            (norm1): BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(336, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
        )
        (transition1): _Transition(
          (norm): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (denseblock2): _DenseBlock(
          (denselayer1): _DenseLayer(
            (norm1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer2): _DenseLayer(
            (norm1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(240, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer3): _DenseLayer(
            (norm1): BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(288, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer4): _DenseLayer(
            (norm1): BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(336, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer5): _DenseLayer(
            (norm1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer6): _DenseLayer(
            (norm1): BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(432, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer7): _DenseLayer(
            (norm1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(480, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer8): _DenseLayer(
            (norm1): BatchNorm2d(528, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(528, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer9): _DenseLayer(
            (norm1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(576, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer10): _DenseLayer(
            (norm1): BatchNorm2d(624, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(624, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer11): _DenseLayer(
            (norm1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(672, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer12): _DenseLayer(
            (norm1): BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(720, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
        )
        (transition2): _Transition(
          (norm): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv): Conv2d(768, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (denseblock3): _DenseBlock(
          (denselayer1): _DenseLayer(
            (norm1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer2): _DenseLayer(
            (norm1): BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(432, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer3): _DenseLayer(
            (norm1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(480, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer4): _DenseLayer(
            (norm1): BatchNorm2d(528, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(528, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer5): _DenseLayer(
            (norm1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(576, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer6): _DenseLayer(
            (norm1): BatchNorm2d(624, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(624, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer7): _DenseLayer(
            (norm1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(672, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer8): _DenseLayer(
            (norm1): BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(720, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer9): _DenseLayer(
            (norm1): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer10): _DenseLayer(
            (norm1): BatchNorm2d(816, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(816, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer11): _DenseLayer(
            (norm1): BatchNorm2d(864, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(864, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer12): _DenseLayer(
            (norm1): BatchNorm2d(912, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(912, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer13): _DenseLayer(
            (norm1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(960, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer14): _DenseLayer(
            (norm1): BatchNorm2d(1008, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1008, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer15): _DenseLayer(
            (norm1): BatchNorm2d(1056, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1056, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer16): _DenseLayer(
            (norm1): BatchNorm2d(1104, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1104, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer17): _DenseLayer(
            (norm1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer18): _DenseLayer(
            (norm1): BatchNorm2d(1200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1200, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer19): _DenseLayer(
            (norm1): BatchNorm2d(1248, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1248, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer20): _DenseLayer(
            (norm1): BatchNorm2d(1296, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1296, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer21): _DenseLayer(
            (norm1): BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1344, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer22): _DenseLayer(
            (norm1): BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1392, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer23): _DenseLayer(
            (norm1): BatchNorm2d(1440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1440, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer24): _DenseLayer(
            (norm1): BatchNorm2d(1488, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1488, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer25): _DenseLayer(
            (norm1): BatchNorm2d(1536, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1536, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer26): _DenseLayer(
            (norm1): BatchNorm2d(1584, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1584, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer27): _DenseLayer(
            (norm1): BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1632, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer28): _DenseLayer(
            (norm1): BatchNorm2d(1680, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1680, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer29): _DenseLayer(
            (norm1): BatchNorm2d(1728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1728, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer30): _DenseLayer(
            (norm1): BatchNorm2d(1776, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1776, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer31): _DenseLayer(
            (norm1): BatchNorm2d(1824, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1824, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer32): _DenseLayer(
            (norm1): BatchNorm2d(1872, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1872, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer33): _DenseLayer(
            (norm1): BatchNorm2d(1920, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1920, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer34): _DenseLayer(
            (norm1): BatchNorm2d(1968, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1968, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer35): _DenseLayer(
            (norm1): BatchNorm2d(2016, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(2016, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer36): _DenseLayer(
            (norm1): BatchNorm2d(2064, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(2064, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
        )
        (transition3): _Transition(
          (norm): BatchNorm2d(2112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv): Conv2d(2112, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (denseblock4): _DenseBlock(
          (denselayer1): _DenseLayer(
            (norm1): BatchNorm2d(1056, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1056, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer2): _DenseLayer(
            (norm1): BatchNorm2d(1104, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1104, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer3): _DenseLayer(
            (norm1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer4): _DenseLayer(
            (norm1): BatchNorm2d(1200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1200, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer5): _DenseLayer(
            (norm1): BatchNorm2d(1248, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1248, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer6): _DenseLayer(
            (norm1): BatchNorm2d(1296, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1296, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer7): _DenseLayer(
            (norm1): BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1344, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer8): _DenseLayer(
            (norm1): BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1392, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer9): _DenseLayer(
            (norm1): BatchNorm2d(1440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1440, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer10): _DenseLayer(
            (norm1): BatchNorm2d(1488, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1488, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer11): _DenseLayer(
            (norm1): BatchNorm2d(1536, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1536, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer12): _DenseLayer(
            (norm1): BatchNorm2d(1584, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1584, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer13): _DenseLayer(
            (norm1): BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1632, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer14): _DenseLayer(
            (norm1): BatchNorm2d(1680, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1680, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer15): _DenseLayer(
            (norm1): BatchNorm2d(1728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1728, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer16): _DenseLayer(
            (norm1): BatchNorm2d(1776, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1776, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer17): _DenseLayer(
            (norm1): BatchNorm2d(1824, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1824, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer18): _DenseLayer(
            (norm1): BatchNorm2d(1872, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1872, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer19): _DenseLayer(
            (norm1): BatchNorm2d(1920, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1920, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer20): _DenseLayer(
            (norm1): BatchNorm2d(1968, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1968, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer21): _DenseLayer(
            (norm1): BatchNorm2d(2016, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(2016, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer22): _DenseLayer(
            (norm1): BatchNorm2d(2064, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(2064, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer23): _DenseLayer(
            (norm1): BatchNorm2d(2112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(2112, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer24): _DenseLayer(
            (norm1): BatchNorm2d(2160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(2160, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
        )
        (norm5): BatchNorm2d(2208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (classifier): Linear(in_features=2208, out_features=1000, bias=True)
    )




```python
# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

    
def create_classifier(model, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the inputb
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
        '''
        # Add the first layer, input to a hidden layer
        layers = [('fc1', nn.Linear(input_size, hidden_layers[0]))]
        layers.append(('relu1', nn.ReLU()))
        layers.append(('dropout1',nn.Dropout(drop_p)))
         
        # Add a variable number of more hidden layers
        hidden_layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        for ii, (h1, h2) in enumerate(hidden_layer_sizes):
            layers.append((''.join(["fc",str(ii+2)]),nn.Linear(h1, h2)))
            layers.append((''.join(["relu",str(ii+2)]),nn.ReLU()))
            layers.append((''.join(["dropout",str(ii+2)]),nn.Dropout(drop_p)))
        
        # Add the last layer
        layers.append((''.join(["fc",str(len(hidden_layers)+1)]),nn.Linear(hidden_layers[-1], output_size)))
        layers.append(('output',nn.LogSoftmax(dim=1)))
    
        classifier = nn.Sequential(OrderedDict(layers))
        print(classifier)
        model.classifier = classifier
    
create_classifier(model, 2208, 102, [1024,256], drop_p=0.5)
model
```

    Sequential(
      (fc1): Linear(in_features=2208, out_features=1024, bias=True)
      (relu1): ReLU()
      (dropout1): Dropout(p=0.5)
      (fc2): Linear(in_features=1024, out_features=256, bias=True)
      (relu2): ReLU()
      (dropout2): Dropout(p=0.5)
      (fc3): Linear(in_features=256, out_features=102, bias=True)
      (output): LogSoftmax()
    )





    DenseNet(
      (features): Sequential(
        (conv0): Conv2d(3, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        (norm0): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu0): ReLU(inplace)
        (pool0): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (denseblock1): _DenseBlock(
          (denselayer1): _DenseLayer(
            (norm1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(96, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer2): _DenseLayer(
            (norm1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(144, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer3): _DenseLayer(
            (norm1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer4): _DenseLayer(
            (norm1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(240, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer5): _DenseLayer(
            (norm1): BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(288, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer6): _DenseLayer(
            (norm1): BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(336, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
        )
        (transition1): _Transition(
          (norm): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (denseblock2): _DenseBlock(
          (denselayer1): _DenseLayer(
            (norm1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer2): _DenseLayer(
            (norm1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(240, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer3): _DenseLayer(
            (norm1): BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(288, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer4): _DenseLayer(
            (norm1): BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(336, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer5): _DenseLayer(
            (norm1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer6): _DenseLayer(
            (norm1): BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(432, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer7): _DenseLayer(
            (norm1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(480, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer8): _DenseLayer(
            (norm1): BatchNorm2d(528, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(528, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer9): _DenseLayer(
            (norm1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(576, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer10): _DenseLayer(
            (norm1): BatchNorm2d(624, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(624, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer11): _DenseLayer(
            (norm1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(672, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer12): _DenseLayer(
            (norm1): BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(720, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
        )
        (transition2): _Transition(
          (norm): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv): Conv2d(768, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (denseblock3): _DenseBlock(
          (denselayer1): _DenseLayer(
            (norm1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer2): _DenseLayer(
            (norm1): BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(432, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer3): _DenseLayer(
            (norm1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(480, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer4): _DenseLayer(
            (norm1): BatchNorm2d(528, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(528, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer5): _DenseLayer(
            (norm1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(576, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer6): _DenseLayer(
            (norm1): BatchNorm2d(624, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(624, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer7): _DenseLayer(
            (norm1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(672, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer8): _DenseLayer(
            (norm1): BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(720, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer9): _DenseLayer(
            (norm1): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer10): _DenseLayer(
            (norm1): BatchNorm2d(816, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(816, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer11): _DenseLayer(
            (norm1): BatchNorm2d(864, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(864, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer12): _DenseLayer(
            (norm1): BatchNorm2d(912, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(912, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer13): _DenseLayer(
            (norm1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(960, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer14): _DenseLayer(
            (norm1): BatchNorm2d(1008, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1008, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer15): _DenseLayer(
            (norm1): BatchNorm2d(1056, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1056, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer16): _DenseLayer(
            (norm1): BatchNorm2d(1104, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1104, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer17): _DenseLayer(
            (norm1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer18): _DenseLayer(
            (norm1): BatchNorm2d(1200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1200, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer19): _DenseLayer(
            (norm1): BatchNorm2d(1248, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1248, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer20): _DenseLayer(
            (norm1): BatchNorm2d(1296, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1296, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer21): _DenseLayer(
            (norm1): BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1344, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer22): _DenseLayer(
            (norm1): BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1392, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer23): _DenseLayer(
            (norm1): BatchNorm2d(1440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1440, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer24): _DenseLayer(
            (norm1): BatchNorm2d(1488, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1488, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer25): _DenseLayer(
            (norm1): BatchNorm2d(1536, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1536, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer26): _DenseLayer(
            (norm1): BatchNorm2d(1584, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1584, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer27): _DenseLayer(
            (norm1): BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1632, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer28): _DenseLayer(
            (norm1): BatchNorm2d(1680, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1680, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer29): _DenseLayer(
            (norm1): BatchNorm2d(1728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1728, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer30): _DenseLayer(
            (norm1): BatchNorm2d(1776, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1776, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer31): _DenseLayer(
            (norm1): BatchNorm2d(1824, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1824, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer32): _DenseLayer(
            (norm1): BatchNorm2d(1872, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1872, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer33): _DenseLayer(
            (norm1): BatchNorm2d(1920, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1920, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer34): _DenseLayer(
            (norm1): BatchNorm2d(1968, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1968, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer35): _DenseLayer(
            (norm1): BatchNorm2d(2016, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(2016, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer36): _DenseLayer(
            (norm1): BatchNorm2d(2064, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(2064, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
        )
        (transition3): _Transition(
          (norm): BatchNorm2d(2112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv): Conv2d(2112, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (denseblock4): _DenseBlock(
          (denselayer1): _DenseLayer(
            (norm1): BatchNorm2d(1056, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1056, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer2): _DenseLayer(
            (norm1): BatchNorm2d(1104, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1104, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer3): _DenseLayer(
            (norm1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer4): _DenseLayer(
            (norm1): BatchNorm2d(1200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1200, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer5): _DenseLayer(
            (norm1): BatchNorm2d(1248, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1248, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer6): _DenseLayer(
            (norm1): BatchNorm2d(1296, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1296, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer7): _DenseLayer(
            (norm1): BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1344, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer8): _DenseLayer(
            (norm1): BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1392, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer9): _DenseLayer(
            (norm1): BatchNorm2d(1440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1440, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer10): _DenseLayer(
            (norm1): BatchNorm2d(1488, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1488, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer11): _DenseLayer(
            (norm1): BatchNorm2d(1536, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1536, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer12): _DenseLayer(
            (norm1): BatchNorm2d(1584, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1584, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer13): _DenseLayer(
            (norm1): BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1632, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer14): _DenseLayer(
            (norm1): BatchNorm2d(1680, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1680, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer15): _DenseLayer(
            (norm1): BatchNorm2d(1728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1728, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer16): _DenseLayer(
            (norm1): BatchNorm2d(1776, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1776, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer17): _DenseLayer(
            (norm1): BatchNorm2d(1824, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1824, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer18): _DenseLayer(
            (norm1): BatchNorm2d(1872, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1872, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer19): _DenseLayer(
            (norm1): BatchNorm2d(1920, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1920, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer20): _DenseLayer(
            (norm1): BatchNorm2d(1968, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1968, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer21): _DenseLayer(
            (norm1): BatchNorm2d(2016, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(2016, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer22): _DenseLayer(
            (norm1): BatchNorm2d(2064, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(2064, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer23): _DenseLayer(
            (norm1): BatchNorm2d(2112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(2112, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer24): _DenseLayer(
            (norm1): BatchNorm2d(2160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(2160, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
        )
        (norm5): BatchNorm2d(2208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (classifier): Sequential(
        (fc1): Linear(in_features=2208, out_features=1024, bias=True)
        (relu1): ReLU()
        (dropout1): Dropout(p=0.5)
        (fc2): Linear(in_features=1024, out_features=256, bias=True)
        (relu2): ReLU()
        (dropout2): Dropout(p=0.5)
        (fc3): Linear(in_features=256, out_features=102, bias=True)
        (output): LogSoftmax()
      )
    )




```python
print(len(dataloaders["trainloader"].batch_sampler),len(dataloaders["validloader"].batch_sampler),len(dataloaders["testloader"].batch_sampler))
```

    103 26 26



```python
def do_validation(model, criterion, validationloader, device):    
    accuracy = 0
    validation_loss = 0
    # Eventhough the model is already on device, move it again to device to avoid problems in case of incosistencies due to code update.
    model.to(device)
    model.eval()
    start = time.time()
    with torch.no_grad():
        for inputs, labels in validationloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.forward(inputs)
            #print("outputs is on cuda", outputs.is_cuda)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
            ps = torch.exp(outputs)
            equality = (labels.data == ps.max(1)[1])
            #print("equality is on cuda", equality.is_cuda)
            if device == 'cuda':
                accuracy += equality.type_as(torch.cuda.FloatTensor()).mean()
            else:
                accuracy += equality.type_as(torch.FloatTensor()).mean()
    #print(f"DEVICE = {device}; Validation time: {(time.time() - start)/3:.3f} seconds")
    model.train()
    return (validation_loss/len(validationloader.batch_sampler)),((100 * accuracy) / len(validationloader.batch_sampler))
```


```python
def do_deep_learning(model, trainloader, epochs, print_every, criterion, optimizer):
    epochs = epochs
    print_every = print_every
    steps = 0

    # change to cuda if possible
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model.to(device)

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            start = time.time()
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            #print(loss.item(),loss.data[0])
            running_loss += loss.item()
            print(f"DEVICE = {device}; Training Time per batch: {(time.time() - start)/3:.3f} seconds")
            if steps % print_every == 0:
                valid_loss, valid_accuracy = do_validation(model, criterion, dataloaders["validloader"], device)
                print("Epoch: {}/{}:".format(e+1, epochs))
                print("Training Loss: {:.4f}".format(running_loss/print_every))
                print("Validation Loss: {:.4f}".format(valid_loss))
                print('Validation Accuracy: %d %%' % valid_accuracy)
                running_loss = 0
                
    
```


```python
def train_model(model, trainloader, learning_rates, nums_epochs, criterions, optimizers):
    for crit in criterions:
        criterion_funct = ''.join(["nn.",crit,"()"])
        print(criterion_funct)
        criterion = eval(criterion_funct)
        for opt in optimizers:
            for lr in learning_rates:
                optimizer_funct = ''.join(["optim.",opt,"(model.classifier.parameters(),lr=",str(lr),")"])
                optimizer = eval(optimizer_funct)
                print(criterion,optimizer) 
                for epochs in nums_epochs:
                    do_deep_learning(model, trainloader, epochs, 50, criterion, optimizer)
                    filepath = ''.join(["checkpoint",str(crit), str(opt),str(epochs)])
                    save_checkpoint(filepath, optimizer, epochs)
    
train_model(model, dataloaders["trainloader"], [0.001], [10], ["NLLLoss"], ["Adam"])
#train_model(model, [0.01,0.001], 2, ["NLLLoss"], ["Adam","SGD"])
```

    nn.NLLLoss()
    NLLLoss() Adam (
    Parameter Group 0
        amsgrad: False
        betas: (0.9, 0.999)
        eps: 1e-08
        lr: 0.001
        weight_decay: 0
    )
    DEVICE = cuda; Training Time per batch: 0.433 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.419 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.410 seconds
    DEVICE = cuda; Training Time per batch: 0.410 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    Epoch: 1/10:
    Training Loss: 0.9040
    Validation Loss: 0.3589
    Validation Accuracy: 91 %
    DEVICE = cuda; Training Time per batch: 0.409 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.410 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.410 seconds
    Epoch: 1/10:
    Training Loss: 0.9020
    Validation Loss: 0.3255
    Validation Accuracy: 92 %
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.165 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.410 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    Epoch: 2/10:
    Training Loss: 0.8090
    Validation Loss: 0.3358
    Validation Accuracy: 91 %
    DEVICE = cuda; Training Time per batch: 0.409 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.410 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.410 seconds
    DEVICE = cuda; Training Time per batch: 0.409 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    Epoch: 2/10:
    Training Loss: 0.9124
    Validation Loss: 0.3394
    Validation Accuracy: 92 %
    DEVICE = cuda; Training Time per batch: 0.410 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.163 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.409 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    Epoch: 3/10:
    Training Loss: 0.7280
    Validation Loss: 0.3054
    Validation Accuracy: 91 %
    DEVICE = cuda; Training Time per batch: 0.409 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.410 seconds
    DEVICE = cuda; Training Time per batch: 0.410 seconds
    DEVICE = cuda; Training Time per batch: 0.408 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.410 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    Epoch: 3/10:
    Training Loss: 0.7724
    Validation Loss: 0.3148
    Validation Accuracy: 91 %
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.163 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.410 seconds
    DEVICE = cuda; Training Time per batch: 0.409 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.409 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    Epoch: 4/10:
    Training Loss: 0.7094
    Validation Loss: 0.3270
    Validation Accuracy: 92 %
    DEVICE = cuda; Training Time per batch: 0.410 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.410 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    Epoch: 4/10:
    Training Loss: 0.7984
    Validation Loss: 0.3180
    Validation Accuracy: 92 %
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.163 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.409 seconds
    DEVICE = cuda; Training Time per batch: 0.410 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    Epoch: 5/10:
    Training Loss: 0.6169
    Validation Loss: 0.3062
    Validation Accuracy: 92 %
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.409 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    Epoch: 5/10:
    Training Loss: 0.7636
    Validation Loss: 0.3001
    Validation Accuracy: 92 %
    DEVICE = cuda; Training Time per batch: 0.409 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.166 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.419 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    Epoch: 6/10:
    Training Loss: 0.5393
    Validation Loss: 0.2991
    Validation Accuracy: 92 %
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.411 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    Epoch: 6/10:
    Training Loss: 0.7801
    Validation Loss: 0.2842
    Validation Accuracy: 92 %
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.164 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.419 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.419 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.419 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.420 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    Epoch: 7/10:
    Training Loss: 0.4769
    Validation Loss: 0.2808
    Validation Accuracy: 92 %
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.419 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    Epoch: 7/10:
    Training Loss: 0.7232
    Validation Loss: 0.3073
    Validation Accuracy: 92 %
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.419 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.419 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.420 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.166 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    Epoch: 8/10:
    Training Loss: 0.3970
    Validation Loss: 0.2715
    Validation Accuracy: 93 %
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    Epoch: 8/10:
    Training Loss: 0.7186
    Validation Loss: 0.2791
    Validation Accuracy: 93 %
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.419 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.419 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.420 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.419 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.420 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.419 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.166 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.419 seconds
    DEVICE = cuda; Training Time per batch: 0.419 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.419 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    Epoch: 9/10:
    Training Loss: 0.3743
    Validation Loss: 0.2728
    Validation Accuracy: 93 %
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.419 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    Epoch: 9/10:
    Training Loss: 0.6738
    Validation Loss: 0.2541
    Validation Accuracy: 92 %
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.419 seconds
    DEVICE = cuda; Training Time per batch: 0.420 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.419 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.163 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.419 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.419 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    Epoch: 10/10:
    Training Loss: 0.3415
    Validation Loss: 0.2840
    Validation Accuracy: 92 %
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.420 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    Epoch: 10/10:
    Training Loss: 0.7289
    Validation Loss: 0.2646
    Validation Accuracy: 93 %
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.413 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.416 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.417 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.418 seconds
    DEVICE = cuda; Training Time per batch: 0.414 seconds
    DEVICE = cuda; Training Time per batch: 0.415 seconds
    DEVICE = cuda; Training Time per batch: 0.412 seconds
    DEVICE = cuda; Training Time per batch: 0.160 seconds


## Testing your network

It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.


```python
# TODO: Do validation on the test set
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
criterion = nn.NLLLoss()
test_loss, test_accuracy = do_validation(model, criterion, dataloaders["testloader"], device)
print("Testing Loss: {:.4f}".format(test_loss))
print('Testing Accuracy: %d %%' % test_accuracy)
```

    Testing Loss: 0.2038
    Testing Accuracy: 94 %


## Save the checkpoint

Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.

```model.class_to_idx = image_datasets['train'].class_to_idx```

Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.


```python
# TODO: Save the checkpoint 
# checkpoint is saved after a model is trained
def save_checkpoint(filepath, optimizer, epochs):
    checkpoint = {'type': model.type,
              'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'class_to_idx': image_datasets["train_data"].class_to_idx,
              'optimizer': optimizer.state_dict(),
              'epochs': epochs, #unclear why it is needed, as it is just a counter
                 }
    torch.save(checkpoint, filepath)
```

## Loading the checkpoint

At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.


```python
# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = eval(''.join(["models.",checkpoint['type'],"(pretrained=True)"]))
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']
    
    return model, optimizer, epochs
```


```python
model, optimizer, epochs = load_checkpoint("checkpointNLLLossAdam10")
```

    /opt/conda/lib/python3.6/site-packages/torchvision-0.2.1-py3.6.egg/torchvision/models/densenet.py:212: UserWarning: nn.init.kaiming_normal is now deprecated in favor of nn.init.kaiming_normal_.



```python
model
```




    DenseNet(
      (features): Sequential(
        (conv0): Conv2d(3, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        (norm0): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu0): ReLU(inplace)
        (pool0): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (denseblock1): _DenseBlock(
          (denselayer1): _DenseLayer(
            (norm1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(96, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer2): _DenseLayer(
            (norm1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(144, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer3): _DenseLayer(
            (norm1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer4): _DenseLayer(
            (norm1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(240, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer5): _DenseLayer(
            (norm1): BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(288, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer6): _DenseLayer(
            (norm1): BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(336, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
        )
        (transition1): _Transition(
          (norm): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (denseblock2): _DenseBlock(
          (denselayer1): _DenseLayer(
            (norm1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer2): _DenseLayer(
            (norm1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(240, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer3): _DenseLayer(
            (norm1): BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(288, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer4): _DenseLayer(
            (norm1): BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(336, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer5): _DenseLayer(
            (norm1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer6): _DenseLayer(
            (norm1): BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(432, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer7): _DenseLayer(
            (norm1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(480, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer8): _DenseLayer(
            (norm1): BatchNorm2d(528, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(528, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer9): _DenseLayer(
            (norm1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(576, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer10): _DenseLayer(
            (norm1): BatchNorm2d(624, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(624, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer11): _DenseLayer(
            (norm1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(672, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer12): _DenseLayer(
            (norm1): BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(720, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
        )
        (transition2): _Transition(
          (norm): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv): Conv2d(768, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (denseblock3): _DenseBlock(
          (denselayer1): _DenseLayer(
            (norm1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer2): _DenseLayer(
            (norm1): BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(432, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer3): _DenseLayer(
            (norm1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(480, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer4): _DenseLayer(
            (norm1): BatchNorm2d(528, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(528, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer5): _DenseLayer(
            (norm1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(576, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer6): _DenseLayer(
            (norm1): BatchNorm2d(624, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(624, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer7): _DenseLayer(
            (norm1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(672, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer8): _DenseLayer(
            (norm1): BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(720, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer9): _DenseLayer(
            (norm1): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer10): _DenseLayer(
            (norm1): BatchNorm2d(816, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(816, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer11): _DenseLayer(
            (norm1): BatchNorm2d(864, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(864, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer12): _DenseLayer(
            (norm1): BatchNorm2d(912, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(912, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer13): _DenseLayer(
            (norm1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(960, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer14): _DenseLayer(
            (norm1): BatchNorm2d(1008, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1008, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer15): _DenseLayer(
            (norm1): BatchNorm2d(1056, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1056, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer16): _DenseLayer(
            (norm1): BatchNorm2d(1104, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1104, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer17): _DenseLayer(
            (norm1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer18): _DenseLayer(
            (norm1): BatchNorm2d(1200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1200, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer19): _DenseLayer(
            (norm1): BatchNorm2d(1248, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1248, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer20): _DenseLayer(
            (norm1): BatchNorm2d(1296, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1296, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer21): _DenseLayer(
            (norm1): BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1344, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer22): _DenseLayer(
            (norm1): BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1392, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer23): _DenseLayer(
            (norm1): BatchNorm2d(1440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1440, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer24): _DenseLayer(
            (norm1): BatchNorm2d(1488, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1488, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer25): _DenseLayer(
            (norm1): BatchNorm2d(1536, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1536, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer26): _DenseLayer(
            (norm1): BatchNorm2d(1584, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1584, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer27): _DenseLayer(
            (norm1): BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1632, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer28): _DenseLayer(
            (norm1): BatchNorm2d(1680, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1680, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer29): _DenseLayer(
            (norm1): BatchNorm2d(1728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1728, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer30): _DenseLayer(
            (norm1): BatchNorm2d(1776, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1776, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer31): _DenseLayer(
            (norm1): BatchNorm2d(1824, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1824, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer32): _DenseLayer(
            (norm1): BatchNorm2d(1872, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1872, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer33): _DenseLayer(
            (norm1): BatchNorm2d(1920, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1920, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer34): _DenseLayer(
            (norm1): BatchNorm2d(1968, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1968, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer35): _DenseLayer(
            (norm1): BatchNorm2d(2016, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(2016, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer36): _DenseLayer(
            (norm1): BatchNorm2d(2064, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(2064, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
        )
        (transition3): _Transition(
          (norm): BatchNorm2d(2112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace)
          (conv): Conv2d(2112, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        (denseblock4): _DenseBlock(
          (denselayer1): _DenseLayer(
            (norm1): BatchNorm2d(1056, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1056, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer2): _DenseLayer(
            (norm1): BatchNorm2d(1104, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1104, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer3): _DenseLayer(
            (norm1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer4): _DenseLayer(
            (norm1): BatchNorm2d(1200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1200, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer5): _DenseLayer(
            (norm1): BatchNorm2d(1248, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1248, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer6): _DenseLayer(
            (norm1): BatchNorm2d(1296, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1296, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer7): _DenseLayer(
            (norm1): BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1344, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer8): _DenseLayer(
            (norm1): BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1392, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer9): _DenseLayer(
            (norm1): BatchNorm2d(1440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1440, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer10): _DenseLayer(
            (norm1): BatchNorm2d(1488, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1488, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer11): _DenseLayer(
            (norm1): BatchNorm2d(1536, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1536, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer12): _DenseLayer(
            (norm1): BatchNorm2d(1584, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1584, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer13): _DenseLayer(
            (norm1): BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1632, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer14): _DenseLayer(
            (norm1): BatchNorm2d(1680, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1680, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer15): _DenseLayer(
            (norm1): BatchNorm2d(1728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1728, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer16): _DenseLayer(
            (norm1): BatchNorm2d(1776, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1776, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer17): _DenseLayer(
            (norm1): BatchNorm2d(1824, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1824, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer18): _DenseLayer(
            (norm1): BatchNorm2d(1872, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1872, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer19): _DenseLayer(
            (norm1): BatchNorm2d(1920, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1920, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer20): _DenseLayer(
            (norm1): BatchNorm2d(1968, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(1968, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer21): _DenseLayer(
            (norm1): BatchNorm2d(2016, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(2016, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer22): _DenseLayer(
            (norm1): BatchNorm2d(2064, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(2064, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer23): _DenseLayer(
            (norm1): BatchNorm2d(2112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(2112, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
          (denselayer24): _DenseLayer(
            (norm1): BatchNorm2d(2160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu1): ReLU(inplace)
            (conv1): Conv2d(2160, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (norm2): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu2): ReLU(inplace)
            (conv2): Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          )
        )
        (norm5): BatchNorm2d(2208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (classifier): Sequential(
        (fc1): Linear(in_features=2208, out_features=1024, bias=True)
        (relu1): ReLU()
        (dropout1): Dropout(p=0.5)
        (fc2): Linear(in_features=1024, out_features=256, bias=True)
        (relu2): ReLU()
        (dropout2): Dropout(p=0.5)
        (fc3): Linear(in_features=256, out_features=102, bias=True)
        (output): LogSoftmax()
      )
    )




```python
optimizer
```




    {'state': {140554128107992: {'step': 103, 'exp_avg': tensor(1.00000e-02 *
              [[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
               [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
               [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
               ...,
               [ 0.0000, -0.0000,  0.0000,  ...,  0.2022,  0.0563,  0.0416],
               [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
               [-0.0000, -0.0000, -0.0000,  ..., -0.0803, -0.2681, -0.0841]], device='cuda:0'), 'exp_avg_sq': tensor(1.00000e-05 *
              [[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
               [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
               [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
               ...,
               [ 0.0000,  0.0000,  0.0000,  ...,  0.3809,  0.1474,  0.1218],
               [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
               [ 0.0000,  0.0000,  0.0000,  ...,  0.3099,  0.1979,  0.0865]], device='cuda:0')},
      140554128108064: {'step': 103, 'exp_avg': tensor(1.00000e-03 *
              [ 0.0000,  0.0000,  0.0000,  ...,  0.7411,  0.0000, -2.1438], device='cuda:0'), 'exp_avg_sq': tensor(1.00000e-06 *
              [ 0.0000,  0.0000,  0.0000,  ...,  2.2310,  0.0000,  2.3374], device='cuda:0')},
      140554128108208: {'step': 103, 'exp_avg': tensor(1.00000e-02 *
              [[ 0.0000,  0.0000,  0.0000,  ..., -0.0336,  0.0000, -0.0994],
               [ 0.0000,  0.0000,  0.0000,  ..., -0.1918,  0.0000,  0.0072],
               [ 0.0000,  0.0000,  0.0000,  ..., -0.0451,  0.0000, -0.2356],
               ...,
               [ 0.0000,  0.0000,  0.0000,  ..., -0.1080,  0.0000, -0.0873],
               [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
               [ 0.0000,  0.0000,  0.0000,  ..., -0.0679,  0.0000,  0.1222]], device='cuda:0'), 'exp_avg_sq': tensor(1.00000e-05 *
              [[ 0.0000,  0.0000,  0.0000,  ...,  0.7568,  0.0000,  0.4262],
               [ 0.0000,  0.0000,  0.0000,  ...,  0.3858,  0.0000,  0.1548],
               [ 0.0000,  0.0000,  0.0000,  ...,  0.6776,  0.0000,  0.3221],
               ...,
               [ 0.0000,  0.0000,  0.0000,  ...,  1.0632,  0.0000,  0.3896],
               [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
               [ 0.0000,  0.0000,  0.0000,  ...,  0.8065,  0.0000,  0.5031]], device='cuda:0')},
      140554128108280: {'step': 103, 'exp_avg': tensor(1.00000e-03 *
              [-0.4258,  0.1768, -1.8711,  0.0000,  2.5771, -3.2059, -0.5527,
               -0.5685, -1.5501,  0.9522, -1.1100, -0.4405, -0.4763, -1.4137,
                0.4520,  0.4673, -1.6163, -1.0456,  0.0000, -0.6026, -0.7771,
                0.0000,  0.0000,  0.4124,  0.0385,  0.0000,  0.5103,  0.9882,
                0.2371,  0.4406, -2.3767, -1.6190, -1.0044,  0.8405,  0.0149,
               -1.1942,  1.4345,  0.0000,  0.0000, -0.6632,  0.2104,  1.8018,
                0.4655,  1.6390, -0.0605, -2.0298, -0.1267,  0.0869,  0.9128,
                0.0020,  0.3988,  0.0000,  0.7991, -1.5292, -0.2992,  1.5077,
                0.0000,  1.0205,  1.7761, -0.0160,  0.6052, -1.9454,  0.2558,
               -0.2615,  0.0000, -0.1023, -1.0845, -2.1324, -0.2900,  0.9261,
                0.0043, -1.8670,  2.4388,  1.5110,  0.0000,  0.0000, -1.7861,
                1.9953,  1.3753,  0.2637, -1.0304, -2.9928, -1.0088,  1.1932,
               -0.6879, -0.2200, -0.1449, -1.3903, -0.4843, -0.4735, -0.6013,
                0.0000,  4.0589,  1.4644, -0.6672,  3.0622, -1.2638, -1.7011,
               -1.4862, -0.9314, -0.0000,  0.0000,  0.6105, -1.5485,  1.2161,
                1.2476,  0.4619, -1.2611,  0.0000,  5.7291, -3.0407,  4.0123,
                2.4374,  0.3236, -2.5899, -0.1135, -1.2039,  0.7763, -0.2409,
                1.4236, -0.8932,  1.0768, -2.6023,  0.0709, -0.9769, -0.0465,
                0.8031, -1.1547,  1.8288, -1.3104, -2.2509, -0.2598, -0.4191,
               -0.9225,  0.0000,  0.5869,  2.6274,  1.6684,  0.1649,  1.1624,
               -0.4364,  2.3582, -1.8659, -1.2063,  0.2196,  0.7846, -0.1462,
               -1.7274,  0.0004, -2.2207,  0.0000, -1.5818,  0.9217,  0.5261,
               -1.1856,  0.3344,  2.2409, -0.5577,  0.5044,  1.9092,  1.0653,
                1.8293, -0.9712, -1.1085,  0.0000,  1.4035, -0.0656, -0.0307,
               -2.0814,  0.0655,  0.9524,  3.2964,  0.9415,  1.5673,  1.8385,
               -0.2182, -0.5502, -0.5320, -1.5166,  0.0000, -0.1769,  0.4973,
                1.4762, -0.9508,  0.4706,  0.0000, -1.8525,  0.0908,  1.8912,
                0.0000,  0.0000, -0.9279, -1.9016,  1.0751,  1.2917, -0.0000,
               -0.3923, -1.5844,  0.0902,  0.6826,  0.3640,  0.6986, -0.6528,
               -0.4003,  0.6920, -0.0613, -0.6312, -3.7640, -0.5014,  0.0000,
                1.3304, -1.6342,  1.7204, -1.1530,  0.0000, -0.3957,  0.1774,
                1.3213, -0.6543, -0.7642, -0.9764, -2.5511,  0.2540,  0.0000,
                0.9077,  0.9776,  0.0085, -1.2823,  0.8302,  0.3202, -0.2141,
               -4.0695,  0.0005, -0.3373, -0.7140,  0.0000,  0.8380, -0.1127,
               -1.0961, -0.7272,  0.0000, -0.0755, -1.0478,  0.4016,  0.8838,
                0.0000,  0.1032,  0.8582,  0.2184, -0.0036,  0.2283,  2.6598,
               -0.3990,  0.0899,  0.0000, -0.7844], device='cuda:0'), 'exp_avg_sq': tensor(1.00000e-06 *
              [ 3.1627,  3.0543,  2.4392,  0.0000,  3.1210,  2.8089,  2.9956,
                3.3246,  3.8742,  4.0046,  2.7098,  3.2769,  1.7790,  5.2325,
                3.1611,  2.0882,  2.5157,  2.2664,  0.0000,  3.4432,  3.1446,
                0.0000,  0.0000,  2.0250,  2.3454,  0.0000,  2.7738,  4.0072,
                2.3085,  3.8454,  3.7365,  4.3308,  3.1828,  2.2727,  0.0330,
                2.8670,  2.6104,  0.0000,  0.0000,  2.7547,  0.1162,  4.1978,
                3.2608,  3.2699,  0.0071,  2.6085,  4.9064,  2.1245,  3.5848,
                0.0034,  2.1838,  0.0000,  3.5795,  3.8151,  3.3446,  3.0618,
                0.0000,  2.9098,  4.1507,  3.7653,  2.5419,  2.7345,  2.9622,
                2.6185,  0.0000,  2.7775,  2.3013,  3.5511,  2.4067,  2.7603,
                0.0003,  3.4862,  5.6833,  2.8351,  0.0000,  0.0000,  2.7070,
                3.4331,  2.7450,  3.6751,  3.2953,  4.6519,  3.3880,  2.7876,
                3.1464,  2.0545,  3.2740,  3.7491,  1.5089,  2.5165,  3.4776,
                0.0000,  3.2175,  2.6474,  2.4833,  2.0096,  3.5134,  2.7188,
                3.2509,  3.5164,  0.0001,  0.0000,  3.7139,  4.1702,  2.6486,
                2.7453,  2.7128,  2.7259,  0.0000,  4.7562,  2.8551,  4.8230,
                2.7052,  2.4412,  3.9187,  0.0025,  3.7330,  2.9104,  3.2760,
                2.6040,  2.7004,  2.1890,  3.5324,  0.0547,  2.3413,  3.5879,
                3.4391,  2.5346,  2.6129,  3.5339,  2.7378,  1.9358,  2.3840,
                2.7179,  0.0000,  2.9256,  3.0779,  3.8738,  2.3995,  2.4485,
                3.2113,  2.5827,  3.4123,  3.6018,  3.5353,  3.2907,  0.0156,
                2.5187,  0.0004,  3.1955,  0.0000,  1.9136,  3.9520,  2.9810,
                2.0837,  2.3963,  3.3489,  0.3974,  0.5922,  3.3185,  3.2819,
                3.1068,  3.3227,  1.6643,  0.0000,  2.8642,  3.7822,  2.8536,
                2.1694,  0.0496,  1.8413,  3.0388,  3.0970,  0.9400,  3.6033,
                3.3166,  2.6341,  2.9621,  2.8592,  0.0000,  2.4798,  3.4014,
                3.8216,  2.1501,  2.3124,  0.0000,  3.3689,  4.6988,  3.6699,
                0.0000,  0.0000,  2.3165,  4.1860,  2.0596,  3.1166,  0.0040,
                0.1905,  3.4100,  2.9807,  2.9934,  2.5642,  2.6738,  1.9011,
                2.4188,  2.1455,  1.9413,  1.8486,  3.1576,  3.6777,  0.0000,
                2.2876,  2.9429,  3.0638,  3.0369,  0.0000,  2.0992,  0.0327,
                2.1728,  3.2852,  2.9928,  2.8583,  3.3317,  1.7746,  0.0000,
                3.1205,  3.0030,  0.0361,  3.7377,  3.6977,  0.1037,  2.6970,
                3.2862,  0.0005,  3.4198,  2.6751,  0.0000,  2.4339,  1.0278,
                2.5936,  2.3220,  0.0000,  0.0193,  3.6892,  2.9108,  2.9304,
                0.0000,  4.7108,  3.5516,  3.5065,  0.0001,  0.8622,  4.9151,
                1.4381,  2.6114,  0.0000,  3.9055], device='cuda:0')},
      140554128108352: {'step': 103, 'exp_avg': tensor(1.00000e-02 *
              [[ 0.2139,  0.2016,  0.1158,  ...,  0.1048,  0.0000,  0.1418],
               [ 0.0426,  0.0131, -0.1147,  ...,  0.1368,  0.0000, -0.0668],
               [-0.4621, -0.2323, -0.2493,  ..., -0.5776,  0.0000,  0.2542],
               ...,
               [ 0.0934,  0.4954,  0.0414,  ...,  0.3099,  0.0000,  0.2812],
               [-0.2536, -0.0805, -1.1545,  ..., -0.7342,  0.0000,  0.0573],
               [ 0.1594, -0.0442,  0.0296,  ..., -0.0609,  0.0000, -0.0305]], device='cuda:0'), 'exp_avg_sq': tensor(1.00000e-04 *
              [[ 0.4028,  0.3136,  0.0826,  ...,  0.0338,  0.0000,  0.0490],
               [ 0.0680,  0.3638,  1.1311,  ...,  0.0293,  0.0000,  0.1332],
               [ 0.2937,  0.3763,  0.1670,  ...,  0.2267,  0.0000,  0.2004],
               ...,
               [ 0.3820,  0.8107,  0.0010,  ...,  0.2127,  0.0000,  0.0453],
               [ 1.0386,  0.0315,  0.3653,  ...,  0.1262,  0.0000,  0.0047],
               [ 0.0091,  0.7419,  0.0004,  ...,  0.1736,  0.0000,  0.4467]], device='cuda:0')},
      140554128108424: {'step': 103, 'exp_avg': tensor(1.00000e-03 *
              [ 1.6869,  0.2191, -1.8937, -0.3707,  0.6984,  0.7684, -0.4692,
               -0.5790, -1.4284, -2.5971,  2.4235,  0.6518, -1.9055, -0.9917,
                0.7050, -1.1592, -0.0207,  0.3742, -2.2119,  2.1146,  1.6739,
                0.8020, -0.0604, -2.3831,  1.4071,  1.1012, -2.2381,  0.2294,
                2.0393, -1.4917,  0.9131,  1.6049,  1.1998,  0.5559, -4.4732,
                1.8644, -4.1409,  0.6022,  2.0939,  1.7997,  3.7994, -4.2037,
                0.4310, -1.7122,  0.1621,  1.6470,  0.7298,  0.6403,  2.1640,
               -1.1688, -2.3075,  1.3309, -1.2332,  0.6626, -1.8659,  1.6018,
                1.7376,  1.6129, -0.6722,  0.0223,  1.2019, -0.3679,  0.0107,
                0.7937,  2.3023, -0.7630,  0.6626,  2.7398, -0.6265,  0.8102,
                0.7509,  3.5346, -2.0135,  1.0395, -2.0295, -2.4820,  2.9909,
               -1.4766, -1.5290,  2.0683,  2.4601,  3.3442, -1.1759, -0.9546,
                0.8742, -1.3713, -3.9226, -2.1445,  0.8342, -0.2638, -0.4368,
                0.6394,  0.8567,  0.0594,  2.0256, -5.7901,  0.7004, -6.5121,
               -1.2396,  1.7789, -0.3554,  0.4835], device='cuda:0'), 'exp_avg_sq': tensor(1.00000e-05 *
              [ 0.4227,  0.3529,  0.3925,  0.7087,  0.2217,  0.9606,  0.4297,
                0.4559,  0.4010,  0.5851,  0.3864,  0.2708,  0.7550,  0.6545,
                0.4208,  0.6650,  0.3407,  0.5850,  0.3966,  0.6119,  0.1320,
                0.4553,  0.1768,  0.6031,  0.5147,  0.6449,  0.7260,  0.6222,
                0.4108,  0.3170,  0.3448,  0.2795,  0.7901,  0.3462,  0.4890,
                0.7527,  1.1667,  0.8463,  0.5180,  0.6574,  1.6490,  0.4148,
                0.4216,  0.8158,  0.2000,  0.4854,  0.2276,  0.3808,  0.4487,
                1.5729,  0.5252,  0.8265,  0.2557,  0.7567,  0.3215,  0.5004,
                0.2698,  0.3232,  0.2477,  0.2212,  0.3047,  0.6201,  0.1441,
                0.2155,  0.6027,  0.2656,  0.5258,  0.5612,  0.3085,  0.4347,
                0.2668,  0.4892,  1.1744,  0.8243,  1.8098,  0.8827,  0.6083,
                0.6100,  0.9772,  0.1971,  0.3691,  0.6208,  0.6289,  0.9992,
                1.0565,  0.9112,  0.6478,  0.6145,  0.7289,  0.9222,  1.0840,
                0.5211,  0.9192,  0.5332,  0.3968,  0.7326,  0.7341,  0.8765,
                1.2991,  0.9295,  0.6862,  0.3595], device='cuda:0')}},
     'param_groups': [{'lr': 0.001,
       'betas': (0.9, 0.999),
       'eps': 1e-08,
       'weight_decay': 0,
       'amsgrad': False,
       'params': [140554128107992,
        140554128108064,
        140554128108208,
        140554128108280,
        140554128108352,
        140554128108424]}]}



# Inference for classification

Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 

```python
probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
```

First you'll need to handle processing the input image such that it can be used in your network. 

## Image Preprocessing

You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 

First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.

Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.

As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 

And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.


```python
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model    
    x = float(image.size[0])
    y = float(image.size[1])
        
    if(x >= y):
        image = image.resize((int(256*(x/y)),256))
    else:
        image = image.resize((256,int(256*(y/x))))

    x = int(image.size[0])
    y = int(image.size[1])
        
    image = image.crop((x/2 - 112,
                       y/2 - 112,
                       x/2 + 112,
                       y/2 + 112))
        

    image = np.array(image)/255
        
    mean = [0.485, 0.456, 0.406]
    stdv = [0.229, 0.224, 0.225]
    image = (image-mean)/stdv 
    image = image.transpose((2, 0, 1))
    return image
```

To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).


```python
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
```


```python
with Image.open('flowers/test/1/image_06760.jpg') as image:
    imshow(process_image(image))
```


![png](output_27_0.png)


## Class Prediction

Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.

To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.

Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.

```python
probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
```


```python
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    inv_class_to_idx = {v: k for k, v in model.class_to_idx.items()}

    with Image.open(image_path) as image:
        image = process_image(image)
        image = torch.FloatTensor([image])

        model.cpu()
        model.eval()  
        with torch.no_grad():
            output = model.forward(image.float())
            top_prob, top_labels = torch.topk(output, topk)
            top_prob = top_prob.exp()  
            top_classes = [inv_class_to_idx[x] for x in top_labels.numpy()[0]]
            return top_prob.numpy()[0], top_classes
```


```python
predict('flowers/test/1/image_06760.jpg',model,5)
```




    (array([ 0.3383702 ,  0.11870196,  0.11778666,  0.11113681,  0.10272884], dtype=float32),
     ['97', '40', '1', '89', '83'])



## Sanity Checking

Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:

<img src='assets/inference_example.png' width=300px>

You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.


```python
# TODO: Display an image along with the top 5 classes
def view_prediction(image_path, model):
    top_prob, top_classes = predict(image_path, model)
    top_classes = [cat_to_name[x] for x in top_classes]
    
    fig, (ax1, ax2) = plt.subplots(figsize=(4,5), nrows=2)
    with Image.open(image_path) as img:
        ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(cat_to_name[image_path.split('/')[2]])
    
    y_pos = np.arange(len(top_classes))
    ax2.barh(y_pos, top_prob)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_classes)
    ax2.invert_yaxis()
```


```python
view_prediction('flowers/test/10/image_07090.jpg',model)
```


![png](output_33_0.png)



```python
view_prediction('flowers/test/20/image_04910.jpg',model)
```


![png](output_34_0.png)

