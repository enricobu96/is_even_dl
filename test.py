from os.path import exists
import torch
from classes.NumberDataset import NumberDataset
import pandas as pd
from utils.performance_measure import precision_recall_f1
import argparse
import pickle

def test(batch_size_test, is_verbose):
    """
    HYPERPARAMETERS AND CONSTANTS
        - BATCH_SIZE_TRAIN: size of the batches for training phase
        - LR: learning rate
        - N_EPOCHS: number of epochs to execute
        - IS_VERBOSE: to avoid too much output
        - WEIGHT_DECAY: the weight decay for the regularization in Adam optimizer
        - USE_VALIDATION: to use the validation set or not. If false only the test set is used, if true the validation set is used
    """
    BATCH_SIZE_TEST=batch_size_test
    IS_VERBOSE=is_verbose

    """
    MODEL INITIALIZATION
    """
    if exists('./models/trained/model.pt'):
        test_set = NumberDataset(pd.read_csv('./datasets/test.csv'))
        test_loader = torch.utils.data.DataLoader(test_set)
        test_size = len(test_set)
        # model = pickle.load(open('./models/trained/model.pt', 'rb'))
        model = torch.load('./models/trained/model.pt')
    else:
        print('Model does not exist, bye')
        exit()
    model.eval()
    loss_function = torch.nn.BCELoss()

    test_loss = 0
    precision = 0
    with torch.no_grad():
        for batch_num, (data, target) in enumerate(test_loader):
            outputs = model(data.float())
            loss = loss_function(outputs, target.float())
            test_loss += loss.item()
            predictions = outputs.data
            predictions = torch.argwhere(predictions)
            target = torch.argwhere(target)

            _precision, _, _ = precision_recall_f1(predictions, target)
            precision += _precision

            if IS_VERBOSE:
                print('Evaluating: Batch %d/%d: Loss: %.4f' % 
                (batch_num, len(test_loader), test_loss / (batch_num + 1)))
        print('TEST PRECISION:', (precision / (test_size/BATCH_SIZE_TEST)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ts', '--batchsizetest', nargs=1, type=int, help='Size of the training batch', required=False)
    parser.add_argument('-v', '--verbose', nargs=1, type=bool, help='Verbose mode on/off', required=False)
    args = parser.parse_args()
    test(
        batch_size_test=args.batchsizetest[0] if args.batchsizetest else 1000,
        is_verbose=args.verbose if args.verbose else True
        )