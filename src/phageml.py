#!/usr/bin/env python3

import math
import pandas as pd
import torch
import torch.cuda
import torch.nn as nn
import torch.optim
import matplotlib.pyplot as plt

from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

## Updated random_split function from a newer version of PyTorch that supports percentages.
def random_split(dataset, lengths,
                 generator=default_generator):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
    ...   ).manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: list[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                pass

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]

alphabet = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]

def one_hot_encode(data, alphabet) -> list:
        tensor_list = []
        for peptide in data:
            t = torch.zeros(len(peptide), 1, len(alphabet), dtype=torch.int8)
            for i, A in enumerate(peptide):
                t[i][0][alphabet.index(peptide[i])] = 1
            tensor_list.append(t)
        return tensor_list

def decode_OHE(output:list) -> list:
    peptide_list = []
    for peptide in output:
        p = str()
        for OHEtensor in peptide:
            pass
    return peptide_list

def return_category(output, categories):
    out_idx = torch.argmax(output).item()
    return categories[out_idx]

class PhageTissueDistributionDataset(Dataset):
    '''Define the the Dataset for Analysis of Phage Distribution in Tissues'''
    def __init__(self, infile: str):
        outfile = pd.read_csv(infile)
        self.x = outfile.iloc[:, 1:]
        self.y = outfile.iloc[:, 0]
        self.data_y = one_hot_encode(self.y, alphabet)
        #self.data_x = torch.tensor(self.x.values)
        self.JGI23_x = torch.tensor(self.split_tissues("JGI23", "1hr", "tpm").values)
    
    def __getitem__(self, idx):
        return self.JGI23_x[idx], self.data_y[idx]
    
    def __len__(self):
        return len(self.data_y)
    
    def split_tissues(self, mouse: str, time: str, outtype: str) -> pd.DataFrame:
        return self.x.loc[:,self.x.columns.str.contains(mouse) & self.x.columns.str.contains(time) & self.x.columns.str.contains(outtype) & self.x.columns.str.endswith("R1")]
    
    def get_categories(self) -> list:
        categories = []
        for item in self.split_tissues("JGI23", "1hr", "tpm"):
            categories.append(item.split(" ")[1]) # type: ignore[call-overload]
        return categories

class PhageRNN(nn.Module):
    '''Defines the Phage RNN Model for Phage Distribution in Tissues'''
    def __init__(self, input_size, hidden_size, output_size):
        super(PhageRNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size+hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size+hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

def main():
    '''main function loop'''
    print("Is PyTorch available: " + str(torch.__version__))
    print("Is CUDA support available: " + str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("ID of the CUDA device: " + str(torch.cuda.current_device()))
        print("Name of the CUDA device: " + str(torch.cuda.get_device_name(torch.cuda.current_device())))
    print("Input file: " + str(args.input))

    ## Define model hyperparameters
    batchSize = 64
    n_hidden = 128
    total_loss = []
    n_iterations = 100
    learning_rate = 0.005

    phageDist = PhageTissueDistributionDataset(args.input)

    ## Split the data into training/testing datasets - 80% training & 20% testing
    train, test = random_split(phageDist, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
    categories = phageDist.get_categories()

    ## Prepare data from training
    train_dataloader = DataLoader(train, batch_size=batchSize, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test, batch_size=batchSize, shuffle=True, drop_last=True)
    
    ## Create PhageRNN model, loss function, and optimizer
    rnn = PhageRNN(len(alphabet), n_hidden, len(categories))    
    loss_func = nn.NLLLoss()
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
    hidden_tensor = rnn.init_hidden()

    ## Create training loop
    for epoch in range(n_iterations):
        for X_batch, Y_batch in train_dataloader:
            for i in range(batchSize): ### NEED TO UPDATE TRAINING LOOP
                y_pred = rnn(X_batch, hidden_tensor)
                loss = loss_func(y_pred, Y_batch)
        
    ## Evaluate training
    rnn.eval()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    args = parser.parse_args()

    main()