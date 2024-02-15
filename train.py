import numpy as np
import pandas as pd
import torch
import warnings; warnings.filterwarnings('ignore')
from classes.NumberDataset import NumberDataset
from utils.performance_measure import precision_recall_f1
from models.EvenNet import EvenNet
import time
# from utils.data_loader import ImageDataset
# from model.cnn import CNN
# import torch.nn as nn
# from torchvision import transforms
# from utils.performance_measure import precision_recall_f1




def execute(batch_size_train=1000, batch_size_test=1000, lr=.5, epochs=2000, patience=5, weight_decay=0.1, model=EvenNet()):
    """
    HYPERPARAMETERS AND CONSTANTS
        - BATCH_SIZE_TRAIN: size of the batches for training phase
        - BATCH_SIZE_TEST: size of the batches for testing phase
        - LR: learning rate
        - N_EPOCHS: number of epochs to execute
        - PATIENCE: the number of previous validation losses smaller than the actual one needed to early stop the training
        - IS_VERBOSE: to avoid too much output
        - WEIGHT_DECAY: the weight decay for the regularization in Adam optimizer
        - USE_VALIDATION: to use the validation set or not. If false only the test set is used, if true the validation set is used
    """
    BATCH_SIZE_TRAIN = batch_size_train
    BATCH_SIZE_TEST = batch_size_test
    LR = lr
    N_EPOCHS = epochs
    PATIENCE = patience
    IS_VERBOSE = True
    WEIGHT_DECAY = weight_decay
    USE_VALIDATION = False

    """
    SETUP
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    classes = ['even', 'odd'] # Get all the classes for one-hot encoding

    def collate_fn(batch):
        return tuple(zip(*batch))

    """
    DATA LOADING
        - Load all data: train, test, validation
    """

    train_set = NumberDataset(pd.read_csv('./datasets/train.csv'))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN)
    test_set = NumberDataset(pd.read_csv('./datasets/test.csv'))
    test_loader = torch.utils.data.DataLoader(test_set)
    if USE_VALIDATION:
        val_set = NumberDataset(pd.read_csv('./datasets/validation.csv'))
        val_loader = torch.utils.data.DataLoader(val_set)

    train_size = len(train_set)
    test_size = len(test_set)
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
        - Uses early stopping if the validation loss does not improve after a certain number of epochs; this depends on PATIENCE.
        - Uses the validation set if USE_VALIDATION is true, otherwise the test set is used
        - The classes are inferred based on the activation threshold
        - Precision, recall and f1 are computed for each epoch and the average is returned
    """
    pre_valid_losses = []
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
            predictions = torch.argwhere(predictions > 0.5)
            target = torch.argwhere(target)
            _precision, _recall, _f1 = precision_recall_f1(predictions, target)
            precision += _precision
            recall += _recall
            f1 += _f1

            if IS_VERBOSE:
                print('Training: Epoch %d - Batch %d/%d: Loss: %.4f' % 
                (epoch, batch_num, len(train_loader), train_loss / (batch_num + 1)))
                

        print('EPOCH', epoch, 'PRECISION:', (precision / (train_size/BATCH_SIZE_TRAIN)))
        print('EPOCH', epoch, 'RECALL:', (recall / (train_size/BATCH_SIZE_TRAIN)))
        print('EPOCH', epoch, 'F1-SCORE:', (f1 / (train_size/BATCH_SIZE_TRAIN)))
        time.sleep(0.5)


    """
    TEST
    """
    test_loss = 0
    precision = 0
    recall = 0
    f1 = 0
    model.eval()
    with torch.no_grad():
        for batch_num, (data, target) in enumerate(test_loader):
            data, target = torch.stack(data, dim=0), torch.stack(target, dim=0)
            outputs = model(data.float())
            loss = loss_function(outputs, target.float())
            test_loss += loss.item()
            predictions = outputs.data
            predictions = torch.argwhere(predictions > 0.5)
            target = torch.argwhere(target)

            _precision, _recall, _f1 = precision_recall_f1(predictions, target)
            precision += _precision
            recall += _recall
            f1 += _f1

            if IS_VERBOSE:
                print('Evaluating: Batch %d/%d: Loss: %.4f' % 
                (batch_num, len(test_loader), test_loss / (batch_num + 1)))
        print('TEST PRECISION:', (precision / (test_size/BATCH_SIZE_TEST)))
        print('TEST RECALL:', (recall / (test_size/BATCH_SIZE_TEST)))
        print('TEST F1-SCORE',  (f1 / (test_size/BATCH_SIZE_TEST)))


execute()