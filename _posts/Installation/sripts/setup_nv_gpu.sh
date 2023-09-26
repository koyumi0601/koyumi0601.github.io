#!/bin/bash
sudo apt update
ubuntu-drivers devices # check recommended graphic driver. 535
sudo apt install nvidia-driver-535 # graphic driver
# cuda toolkit old
# sudo apt-get install nvidia-cuda-toolkit # cuda
# cuda toolkit 11.8 start
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
# cuda toolkit 11.8 end
pip install tensorflow
pip install torch # CPU & GPU pytorch
pip install torchvision # image, video preprocessing
pip install matplotlib
pip install numpy
# cudnn download and move to cuda toolkit location

