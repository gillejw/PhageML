#!/usr/bin/env python3

import math
import numpy as np
import pandas as pd
import re
import torch
import torch.cuda

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

class PhageTissueDistributionDataset(Dataset):
    '''Define the the Dataset for Analysis of Phage Distribution in Tissues'''
    def __init__(self, infile: str):
        outfile = pd.read_csv(infile)
        self.x = outfile.iloc[:, 1:]
        self.data_y = outfile.iloc[:, 0].values
        #self.data_x = torch.tensor(self.x.values)
        self.JGI23_x = torch.tensor(self.split_tissues("JGI23", "1hr", "tpm").values)
    
    def __getitem__(self, idx):
        return self.JGI23_x[idx], self.data_y[idx]
    
    def __len__(self):
        return len(self.data_y)
    
    def split_tissues(self, mouse, time, outtype):
        return self.x.loc[:,self.x.columns.str.contains(mouse) & self.x.columns.str.contains(time) & self.x.columns.str.contains(outtype) & self.x.columns.str.endswith("R1")]

def main():
    '''main function loop'''
    print("Is PyTorch available: " + str(torch.__version__))
    print("Is CUDA support available: " + str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("ID of the CUDA device: " + str(torch.cuda.current_device()))
        print("Name of the CUDA device: " + str(torch.cuda.get_device_name(torch.cuda.current_device())))
    print("Input file: " + str(args.input))

    phageDist = PhageTissueDistributionDataset(args.input)

    ## Split the data into training/testing datasets - 80% training & 20% testing
    train, test = random_split(phageDist, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

    ## Prepare data from training
    train_dataloader = DataLoader(train, batch_size=64, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test, batch_size=64, shuffle=True, drop_last=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    args = parser.parse_args()

    main()