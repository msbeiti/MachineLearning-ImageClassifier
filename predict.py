#!/usr/bin/env python3
import json
import os
import time
import warnings

import numpy as np
import torch
from PIL import Image
from torch import nn, optim

import utilities
from torchvision import models


def load_model(filepath, device, logger):
    try:
        if device == 'cuda':
            checkpoint = torch.load(filepath)
        else:
             checkpoint = torch.load(filepath, map_location='cpu')
        model = eval(''.join(["models.",checkpoint['type'],"(pretrained=True)"]))
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        optimizer = checkpoint['optimizer']
        epochs = checkpoint['epochs']
        logger.critical("Successfully loading model.\n %s" % model)

        return model, optimizer, epochs
    except Exception as e:
        logger.critical("Could not load model.\n %s" % e)
        exit(1)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

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


def predict(image_path, model, topk, device, logger):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    inv_class_to_idx = {v: k for k, v in model.class_to_idx.items()}
    try:
        with Image.open(image_path) as image:
            image = process_image(image)
            if device == 'cuda':
                image = torch.cuda.FloatTensor([image])
            else:
                image = torch.FloatTensor([image])
            model.eval()
            model.to(device)
            with torch.no_grad():
                    output = model.forward(image.float())
                    top_prob, top_labels = torch.topk(output, topk)
                    top_prob = top_prob.exp()
                    top_classes = [inv_class_to_idx[x] for x in top_labels.cpu().numpy()[0]]
                    return top_prob.cpu().numpy()[0], top_classes
    except Exception as e:
        logger.critical("Could not perform prection.\n %s" % e)
        exit(1)



if __name__ == "__main__":
    '''The entrypoint of the image classifier predict module'''
    # ToDo function needs optimization and a language consistency check.
    warnings.filterwarnings("ignore")
    args, logger = utilities.parse_predict_arguments()

    if not os.path.isfile(args.image_file):
        logger.error("Image file \"%s\" not found." % args.image_file)
        exit(1)
    if not os.path.isfile(args.model_file):
        logger.error("Model checkpoint file \"%s\" not found." % args.model_file)
        exit(1)
    if not os.path.isfile(args.categories_json):
        logger.error("categories_json file \"%s\" (mapping categories to name) not found." % args.categories_json)
        exit(1)

    if args.gpu:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            logger.critical("cuda/GPU is not available")
            exit(1)
    else:
        device = 'cpu'

    # load categories
    with open(args.categories_json, 'r') as f:
            cat_to_name = json.load(f)

    # load model
    model, optimizer, epochs = load_model(args.model_file, device, logger)
    top_probs, top_classes = predict(args.image_file, model,args.top_k, device, logger)

    predictions = list(zip(top_classes, top_probs))
    print(f"\nPredicted result:")
    logger.info(f"\nPredicted result:")
    for i in range(len(predictions)):
            print('{} : {:.3%}'.format(cat_to_name[predictions[i][0]], predictions[i][1]))
            logger.info('{} : {:.3%}'.format(cat_to_name[predictions[i][0]], predictions[i][1]))
    print(f"\nExpected result: {cat_to_name[args.image_file.split('/')[-2]]}\n")
    logger.info(f"\nExpected result: {cat_to_name[args.image_file.split('/')[-2]]}\n")
