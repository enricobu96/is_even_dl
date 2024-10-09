from os.path import exists
import torch
from classes.NumberDataset import NumberDataset
import pandas as pd
from utils.performance_measure import precision_recall_f1
import argparse
from models.EvenNet import EvenNet


def is_even(num):
    """
    MODEL INITIALIZATION
    """
    model = EvenNet()
    if exists('./models/trained/model.pt'):
        model.load_state_dict(torch.load('./models/trained/model.pt', weights_only=True))
    else:
        print('Model does not exist, bye')
        exit()
    model.eval()

    with torch.no_grad():
        outputs = model(torch.tensor([num]).float())
        predictions = outputs.data
        print(predictions)
        print('even' if predictions[0]>0 else 'odd')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('number', type=int)
    args = parser.parse_args()
    is_even(args.number)