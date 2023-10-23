#!/usr/bin/env python3

import torch.cuda

print("Is CUDA support available: " + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("ID of the CUDA device: " + str(torch.cuda.current_device()))
    print("Name of the CUDA device: " + str(torch.cuda.get_device_name(torch.cuda.current_device())))

