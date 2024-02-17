from os.path import exists
import torch
from classes.NumberDataset import NumberDataset
import pandas as pd
from utils.performance_measure import precision_recall_f1
import argparse
import pickle

def is_even(num):
    """
    MODEL INITIALIZATION
    """
    if exists('./models/trained/model.pt'):
        model = torch.load('./models/trained/model.pt')
    else:
        print('Model does not exist, bye')
        exit()
    model.eval()

    sample = {'number': num, 'label': 'even'}
    sample = pd.DataFrame(data=sample, index=[0])
    sample = NumberDataset(sample)
    sample_loader = torch.utils.data.DataLoader(sample)

    precision = 0
    with torch.no_grad():
        for batch_num, (data, target) in enumerate(sample_loader):
            outputs = model(data.float())
            predictions = outputs.data
            predictions = torch.argwhere(predictions)
            print(predictions)
            print('even' if predictions[0]>0.5 else 'odd')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('number', type=int)
    args = parser.parse_args()
    is_even(args.number)