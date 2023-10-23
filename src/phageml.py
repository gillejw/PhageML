#!/usr/bin/env python3

import torch
import torch.cuda

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

class PhageTissueDistributionDataset(Dataset):
    '''Define the the Dataset for Analysis of Phage Distribution in Tissues'''
    def __init__(self,):
        self.aaa

    def __len__(self):
        return len(XXX)
    
    def __getitem__(self, idx):
        sss

def main():
    '''main function loop'''
    print("Is PyTorch available: " + str(torch.__version__))
    print("Is CUDA support available: " + str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("ID of the CUDA device: " + str(torch.cuda.current_device()))
        print("Name of the CUDA device: " + str(torch.cuda.get_device_name(torch.cuda.current_device())))
    print("Input file: " + str(args.input))

    #phageDist = PhageTissueDistributionDataset()

    train, test = random_split(range(30), [10, 20], generator=torch.Generator().manual_seed(42))

    ## Prepare data from training
    train_dataloader = DataLoader(train, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=64, shuffle=True)

    for item in train_dataloader:
        print(item)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    args = parser.parse_args()

    main()