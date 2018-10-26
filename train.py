#!/usr/bin/env python3
import os
import time
from collections import OrderedDict
import warnings

import torch
from torch import nn, optim
from torchvision import models

import loggers
import utilities




def create_classifier(model, input_size, output_size, hidden_layers, drop_p, logger):
        ''' Builds a classifier with arbitrary hidden layers.

            Arguments
            ---------
            input_size: integer, size of the inputb
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
            logger: log important iformation
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
        model.classifier = classifier
        logger.info("The following classifier has been created %s." % model.classifier)

def do_validation(model, criterion, validationloader, device, logger):
    accuracy = 0
    validation_loss = 0
    model.to(device)
    model.eval()
    start = time.time()
    with torch.no_grad():
        for inputs, labels in validationloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
            ps = torch.exp(outputs)
            equality = (labels.data == ps.max(1)[1])
            if device == 'cuda':
                accuracy += equality.type_as(torch.cuda.FloatTensor()).mean()
            else:
                accuracy += equality.type_as(torch.FloatTensor()).mean()
    model.train()
    return (validation_loss/len(validationloader.batch_sampler)),((100 * accuracy) / len(validationloader.batch_sampler))


def do_deep_learning(model, trainloader, epochs, print_every, criterion, optimizer, has_gpu, logger):
    epochs = epochs
    print_every = print_every
    steps = 0

    # change to cuda if possible
    if has_gpu:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            logger.critical("cuda/GPU is not available")
            exit(1)
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
            running_loss += loss.item()
            print(f"DEVICE = {device}; Training Time per batch: {(time.time() - start)/3:.3f} seconds")
            logger.info(f"DEVICE = {device}; Training Time per batch: {(time.time() - start)/3:.3f} seconds")
            if steps % print_every == 0:
                valid_loss, valid_accuracy = do_validation(model, criterion, dataloaders["validloader"], device, logger)
                print("Epoch: {}/{}:".format(e+1, epochs))
                logger.info("Epoch: {}/{}:".format(e+1, epochs))
                print("Training Loss: {:.4f}".format(running_loss/print_every))
                logger.info("Training Loss: {:.4f}".format(running_loss/print_every))
                print("Validation Loss: {:.4f}".format(valid_loss))
                logger.info("Validation Loss: {:.4f}".format(valid_loss))
                print("Validation Accuracy: %d %%" % valid_accuracy)
                logger.info("Validation Accuracy: %d %%" % valid_accuracy)
                running_loss = 0


def train_save_model(model, trainloader, learning_rates, nums_epochs, criterions, optimizers, has_gpu, save_dir, logger):
    # Training function support a list of learning rates, epochs and criterions
    # TODO extend it to return the best combination of values from the aforementioned lists
    print_every = len(trainloader.batch_sampler)/4
    for crit in criterions:
        criterion_funct = ''.join(["nn.",crit,"()"])
        logger.info("Current criterion function. \n %s" % criterion_funct)
        criterion = eval(criterion_funct)
        for opt in optimizers:
            for lr in learning_rates:
                optimizer_funct = ''.join(["optim.",opt,"(model.classifier.parameters(),lr=",str(lr),")"])
                optimizer = eval(optimizer_funct)
                logger.info("Current optimizer function. \n %s" % optimizer)
                for epochs in nums_epochs:
                    print("The training of the model has started, this can take a while, you might get a coffee in the meantime...\n")
                    do_deep_learning(model, trainloader, epochs, print_every, criterion, optimizer, has_gpu, logger)
                    filepath = ''.join([save_dir,"checkpoint",str(crit), str(opt),str(epochs)])
                    save_model(filepath, optimizer, epochs,logger)

def save_model(filepath, optimizer, epochs, logger):
    checkpoint = {'type': model.type,
              'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'class_to_idx': image_datasets["train_data"].class_to_idx,
              'optimizer': optimizer.state_dict(),
              'epochs': epochs, #unclear why it is needed, as it is just a counter
                 }
    try:
        torch.save(checkpoint, filepath)
        logger.info("Model was successfully saved %s" % filepath)
    except Exception as e:
        logger.critical("Could not save model.\n %s" % e)


if __name__ == "__main__":
    '''The entrypoint of the image classifier train module'''
    warnings.filterwarnings("ignore")
    args, logger = utilities.parse_train_arguments()

    if not os.path.isdir(args.data_directory):
       logger.error("Data directory \"%s\" not found." % args.data_directory)
       exit(1)

    if args.save_dir:
        if not os.path.isdir(args.save_dir):
            logger.info("Checkpoint directory \"%s\" not found." % args.save_dir)
            logger.info("Creating checkpoint directory \"%s\"." % args.save_dir)
            os.makedirs(args.save_dir)

    try:
        hidden_units = [int(x) for x in args.hidden_units.split(",")]
        assert isinstance(hidden_units, list)
    except:
        logger.error("hidden_units argument MUST be a python list")

    # Load the images data
    dataloaders, image_datasets = utilities.load_images(args.data_directory, logger)

    # Create a pre-trained model on images features and the disable any change on its parameters
    model = eval(''.join(["models.",args.arch,"(pretrained=True)"]))
    model.type = args.arch
    for param in model.parameters():
        param.requires_grad = False

    # Derive input_size
    if isinstance(model.classifier, nn.Sequential):
        input_size = list(model.classifier.children())[0].in_features
    else:
        input_size = model.classifier.in_features

    # Derive output_size
    output_size = len(image_datasets['train_data'].classes)

    # Create an appropriate classifier for the data and add it to the model
    create_classifier(model, input_size, output_size, hidden_units, args.dropout_probability, logger)
    logger.info("The following model has been created %s." % model)
    print("\n\n Thank you for using this image classifier application.\n")
    print("The following model has been created for you %s." % model)

    # Train, validate and save the model
    train_save_model(model, dataloaders["trainloader"], [args.learning_rate], [args.epochs], ["NLLLoss"], ["Adam"], args.gpu, args.save_dir,logger)

    # Test the model on the testing set - giving a final feedback how good the accuracy of the model is/was
    if args.gpu and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    _, test_accuracy = do_validation(model, nn.NLLLoss(), dataloaders["testloader"], device, logger)
    print('Model accuracy on testing set: %d %%' % test_accuracy)
    logger.info('Model accuracy on testing set: %d %%' % test_accuracy)
