# is_even_dl

This project aims to build a neural network which classifies a number as even or odd. If this sounds extremely stupid: it is. But it becomes interesting when you consider it from the xDNN (explainable deep neural network) perspective: we, in fact, already know what the function our neural network is trying to generalize is. The function is the **modulo** function, defined as follow:
```
f(x)=mod(x,2)
```
This is interesting because on one hand is a really simple function, while on the other hand it's a discontinuous function, so it needs some kind of non-linearity (e.g. a relu function) to be correctly approximized. Hence, the use of a neural network. Note that the function should be in the real domain, but we consider it as a discrete function where input can only be an integer number and the output is *0* or *1*.

Since we know the input and the function our neural network is trying to optimize, we can start studying the single nodes in the NN and try to understand why we get a certain result (still working on this).

## Usage

### Preparation

First of all, you need to install the requirements for the project. Just use pip to install the needed libraries:
```
pip install -r requirements.txt
```

### Synthetic dataset generation 

First of all, you need to generate a synthetic dataset. This can be easily achived with the `utils/generate_dataset.py` script; use it like this:
```
python3 generate_dataset.py -b batch_size -s size -p precision -t trainsize -v validation
```
where:
- _batch_size_ is the only mandatory argument, and as the name suggests is the batch size you want to use for your training
- _size_ is the size of the dataset you want to generate (train+test+possibly validation sets)
- _precision_ is the range within the random discrete numbers are generated. Allowed precisions are 8, 16, 32, 64 (so [-2^8, 2^8], [-2^16, 2^16] and so on)
- _trainsize_ is the percentage of training set size against test/validation sets
- _validation_ is a boolean: true if we want to use validation during training, false otherwise

### Training

After you created the synthetic dataset, you can start training your neural network. To do so, use the `train.py` script in the following way:
```
python3 train.py 
```