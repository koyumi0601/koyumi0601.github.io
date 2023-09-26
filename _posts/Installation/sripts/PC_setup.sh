#!/bin/bash
echo "Install Chrome..."
echo "**************************************************************************"
sudo sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
sudo apt update 
sudo apt install google-chrome-stable
google-chrome-stable 
echo "Install Chrome Complete!"
echo "**************************************************************************"
echo "Install Github Desktop..."
echo "**************************************************************************"
sudo apt install git
sudo apt-get update
sudo apt-get install snapd
wget https://github.com/shiftkey/desktop/releases/download/release-2.0.4-linux1/GitHubDesktop-linux-2.0.4-linux1.snap
sudo snap install GitHubDesktop-linux-2.0.4-linux1.snap --classic --dangerous
echo "Install Github Desktop Complete..."
echo "**************************************************************************"
echo "Install VS Code..."
echo "**************************************************************************"
sudo apt update
sudo apt install software-properties-common apt-transport-https wget
wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"
sudo apt install code
echo "Install VS Code Complete..."
echo "**************************************************************************"
echo "Install VS Python3..."
echo "**************************************************************************"
sudo apt install python3-pip

echo "Install VS Python3 Complete..."
echo "**************************************************************************"
echo "Install VS Typora..."
echo "**************************************************************************"
wget -qO - https://typora.io/linux/public-key.asc | sudo tee /etc/apt/trusted.gpg.d/typora.asc
sudo add-apt-repository 'deb https://typora.io/linux ./'
sudo apt-get update
sudo apt-get install typora
echo "Install VS Typora Complete..."
echo "**************************************************************************"
echo "Install VS Flameshot..."
echo "**************************************************************************"
sudo apt update 
sudo apt install flameshot
echo "Install VS Flameshot Complete..."
echo "**************************************************************************"
echo "Install VS Docker..."
echo "**************************************************************************"
sudo apt update # update package manager
sudo apt install apt-transport-https ca-certificates curl software-properties-common # install required package
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg # Add official GPG key of docker
echo "deb [signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null # Add docker repository
sudo apt update # update package manager again
sudo apt install docker-ce # install docker
sudo systemctl start docker # start docker
sudo systemctl enable docker # set docker start when bootup
echo "Install VS Docker Complete..."
echo "**************************************************************************"
echo "**************************************************************************"
echo "**************************************************************************"
echo "Chrome, Github Desktop, VS code, Python3, Typora, Flameshot, Docker"
echo "**************************************************************************"
echo "**************************************************************************"
echo "**************************************************************************"
echo "NVIDIA GPU related.."
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
echo "**************************************************************************"
echo "**************************************************************************"
echo "**************************************************************************"
echo "cudnn download and move to cuda toolkit location.."
# cudnn download and move to cuda toolkit location


