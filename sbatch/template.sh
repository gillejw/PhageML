#!/bin/bash
#SBATCH --ntasks=1                                  ## total number of tasks across all nodes
#SBATCH --mem=5G                                    ## RAM per node in Gb
#SBATCH --time=01:00:00                             ## total run time limit (HH:MM:SS)
#SBATCH --output=/scratch/gillejw/job-%j.out        ## output file, where %j is replaced with the job ID
#SBATCH --error=/scratch/gillejw/job-%j.err         ## error file, where %j is replaced with the job ID
#SBATCH --partition=gpu2                            ## sets the gpu partition, where the the number of GPUs per device are present on the node (gpu2, gpu4)
#SBATCH --gres=gpu:tesla:1                          ## total number of GPU devices to request

module load python/3.9.2
module load cuda11.0/toolkit/11.0.3
source /home/gillejw/coding/PhageML/.env/bin/activate

# Install previous version of PyTorch (v1.7.1) to be compatible with cuda 11.0.4 -- Need to update to new NVIDIA CUDA Toolkit!
pip install --quiet torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

pip3 install --user -r /home/gillejw/coding/PhageML/requirements.txt

python /home/gillejw/coding/PhageML/src/hello.py