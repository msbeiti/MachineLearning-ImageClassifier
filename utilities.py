#!/usr/bin/env python3
import argparse
import loggers
import torch
from torchvision import datasets, transforms

def parse_train_arguments():
            '''
                Basic usage: python train.py data_directory
                Prints out training loss, validation loss, and validation accuracy as the network trains
                Options:
                    Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
                    Choose architecture: python train.py data_dir --arch "vgg13"
                    Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
                    Use GPU for training: python train.py data_dir --gpu
            '''
            parser = argparse.ArgumentParser(
                description='Image Classifier Train Module')
            parser.add_argument('data_directory', action="store")
            parser.add_argument('--save_dir', type=str, default='checkpoints/',
                                help='path to saved checkpoints')
            parser.add_argument('--arch', type=str, choices=['densenet161', 'vgg16'],
                                default='densenet161', help='choose between densenet161 and vgg16')
            parser.add_argument('--learning_rate', type=float,
                                default=0.001, help='learning_rate')
            parser.add_argument('--dropout_probability', type=float,
                                            default=0.5, help='dropout_probability')
            parser.add_argument('--hidden_units', type=str,
                                default="1024,256", help='comma separated hidden_units, e.g., 1024,512,256')
            parser.add_argument('--epochs', type=int,
                                default=10, help='number of epochs')
            parser.add_argument('--gpu', default=False, action='store_true', help='GPU processing enabler')
            parser.add_argument('--logger', default='console',choices=['console', 'file'], help="log to console or to the file")

            args = parser.parse_args()
            logger = loggers.create_logger(name="imageclassifier_train", type=args.logger)
            logger.debug(args)
            return args, logger

def parse_predict_arguments():
            '''
            Basic usage: python predict.py /path/to/image checkpoint
            Options:
            Return top KKK most likely classes: python predict.py input checkpoint --top_k 3
            Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
            Use GPU for inference: python predict.py input checkpoint --gpu
                '''
            parser = argparse.ArgumentParser(
                description='Image Classifier Predict Module')
            parser.add_argument('image_file', action="store", help='path to image file')
            parser.add_argument('model_file', action="store", help='path to saved model file')
            parser.add_argument('--top_k', type=float, default=5, help='number of returned most likely classes')
            parser.add_argument('--categories_json', type=str, default="cat_to_name.json", help='path to mapping categories to names')
            parser.add_argument('--gpu', default=False, action='store_true', help='GPU processing enabler')
            parser.add_argument('--logger', default='console',choices=['console', 'file'], help="log to console or to the file")
            args = parser.parse_args()
            logger = loggers.create_logger(name="imageclassifier_predict", type=args.logger)
            logger.debug(args)
            return args, logger

def load_images(data_directory, logger):
    data_dir = data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

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
    try:
        image_datasets = {'train_data': datasets.ImageFolder(train_dir, transform=data_transforms['train_transforms']),
                     'valid_data': datasets.ImageFolder(valid_dir, transform=data_transforms['valid_test_transforms']),
                      'test_data': datasets.ImageFolder(test_dir, transform=data_transforms['valid_test_transforms'])}

        dataloaders = {'trainloader' : torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=64, shuffle=True),
                   'validloader' : torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=32),
                    'testloader' : torch.utils.data.DataLoader(image_datasets['test_data'], batch_size=32)}

        logger.info("Images were successfully loaded")
    except Exception as e:
        logger.critical("Could not load images.\n %s" % e)
        exit(1)

    return dataloaders, image_datasets
