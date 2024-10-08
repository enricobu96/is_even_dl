import numpy as np
import pandas as pd
import torch
import warnings; warnings.filterwarnings('ignore')
from classes.NumberDataset import NumberDataset
from utils.performance_measure import precision_recall_f1
from models.EvenNet import EvenNet
import time
import argparse
import pickle

def train(batch_size_train, lr, epochs, is_verbose, weight_decay, use_validation):
    """
    HYPERPARAMETERS AND CONSTANTS
        - BATCH_SIZE_TRAIN: size of the batches for training phase
        - LR: learning rate
        - N_EPOCHS: number of epochs to execute
        - IS_VERBOSE: to avoid too much output
        - WEIGHT_DECAY: the weight decay for the regularization in Adam optimizer
        - USE_VALIDATION: to use the validation set or not. If false only the test set is used, if true the validation set is used
    """
    BATCH_SIZE_TRAIN = batch_size_train
    LR = lr
    N_EPOCHS = epochs
    IS_VERBOSE = is_verbose
    WEIGHT_DECAY = weight_decay
    USE_VALIDATION = use_validation

    """
    SETUP
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    """
    DATA LOADING
        - Load all data: train, test, validation
    """

    train_set = NumberDataset(pd.read_csv('./datasets/train.csv'))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN)

    if USE_VALIDATION:
        val_set = NumberDataset(pd.read_csv('./datasets/validation.csv'))
        val_loader = torch.utils.data.DataLoader(val_set)

    train_size = len(train_set)

    model = EvenNet(input_dim=batch_size_train)
    """
    MODEL INITIALIZATION
        - optimizer: Adam with weight decay as regularization technique
        - loss function: binary cross entropy loss
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_function = torch.nn.MSELoss()

    """
    TRAIN
        Notes:
        - Uses the validation set if USE_VALIDATION is true, otherwise the test set is used
        - The classes are inferred based on the activation threshold
        - Precision, recall and f1 are computed for each epoch and the average is returned
    """
    for epoch in range(N_EPOCHS):
        train_loss = 0
        valid_losses = []
        precision = 0
        recall = 0
        f1 = 0

        for batch_num, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data.float())
            loss = loss_function(outputs, target.float())

            loss.backward()
            optimizer.step()
            train_loss += loss.item()


            predictions = outputs.data
            predictions = torch.argwhere(predictions < 0)
            target = torch.argwhere(target > 0)
            _precision, _, _ = precision_recall_f1(predictions, target)
            precision += _precision

            if IS_VERBOSE:
                print('Training: Epoch %d - Batch %d/%d: Loss: %.4f' % 
                (epoch, batch_num, len(train_loader), train_loss / (batch_num + 1)))
                

        print('EPOCH', epoch, 'PRECISION:', (precision / (train_size/BATCH_SIZE_TRAIN)))
        time.sleep(2)

    # pickle.dump(model, open('./models/trained/model.pt', 'wb'))
    torch.save(model, './models/trained/model.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ts', '--batchsizetrain', nargs=1, type=int, help='Size of the training batch', required=False)
    parser.add_argument('-lr', '--learningrate', nargs=1, type=float, help='Learning rate', required=False)
    parser.add_argument('-e', '--epochs', nargs=1, type=int, help='Number of epochs', required=False)
    parser.add_argument('-v', '--verbose', nargs=1, type=bool, help='Verbose mode on/off', required=False)
    parser.add_argument('-wd', '--weightdecay', nargs=1, type=float, help='Weight decay (L2 regularization)', required=False)
    parser.add_argument('-va', '--validation', nargs=1, type=bool, help='Wether use or not validation set', required=False)
    args = parser.parse_args()
    train(
        batch_size_train=args.batchsizetrain[0] if args.batchsizetrain else 1000,
        lr=args.learningrate[0] if args.learningrate else 0.5,
        epochs=args.epochs[0] if args.epochs else 10,
        is_verbose=args.verbose if args.verbose else True,
        weight_decay=args.weightdecay[0] if args.weightdecay else 0.9,
        use_validation=args.validation
        )