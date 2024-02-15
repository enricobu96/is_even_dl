import sys
import random
from enum import Enum
import argparse
from tqdm import tqdm
import threading


def generate(args):
    # Calculate parameters
    size = args.size[0] if args.size else 100000
    precision = args.precision[0] if args.precision else 32
    trainsize = args.trainsize[0] if args.trainsize else 0.8
    validation = args.validation
    testsize = 1-trainsize if not validation else (1-trainsize)*.5
    minv, maxv = -(2**precision), 2**precision

    # Label function
    ef = lambda x: 'even' if x%2==0 else 'odd'

    print('Generating random dataset, brb...')
    def generate_train():
        with open("../datasets/train.csv", 'w') as f:
            print('Generating training set')
            for i in tqdm(range(int(trainsize*size)), colour="green"):
                n = random.randint(minv, maxv)
                f.write(str(n) + ',' + ef(n) + '\n')

    def generate_test():
        with open("../datasets/test.csv", 'w') as f:
            print('Generating test set')
            for i in tqdm(range(int(testsize*size)+1), colour="magenta"):
                n = random.randint(minv, maxv)
                f.write(str(n) + ',' + ef(n) + '\n')

    def generate_validation():
        with open("../datasets/validation.csv", 'w') as f:
            print('Generating validation set')
            for i in tqdm(range(int(testsize*size)+1), colour="blue"):
                n = random.randint(minv, maxv)
                f.write(str(n) + ',' + ef(n) + '\n')
    
    # Threading because yes
    t1 = threading.Thread(target=generate_train)
    t2 = threading.Thread(target=generate_test)
    t3 = threading.Thread(target=generate_validation)

    t1.start()
    t2.start()
    if validation: t3.start()

    t1.join()
    t2.join()
    if validation: t3.join()
    
    print('Done, kthxbye!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--size', nargs=1, type=int, help='Size of the dataset to generate', required=False)
    parser.add_argument('-p', '--precision', nargs=1, type=int, choices=[8, 16, 32, 64], help='Precision of the dataset to generate', required=False)
    parser.add_argument('-t', '--trainsize', nargs=1, type=float, choices=[i/10 for i in range(1, 10)], help='Size of training set (0.1 to 1)', required=False)
    parser.add_argument('-v', '--validation', nargs=1, type=bool, help='Wether generate or not validation set', required=False)
    args = parser.parse_args()
    generate(args)