# SCALABLE NEURAL QUANTUM STATES ARCHITECTURE FOR QUANTUM CHEMISTRY #

Tianchen Zhao, Giuseppe Carleo, James Stokes and Shravan Veerapaneni

This repository includes the codes for the paper "Natural Evolution Strategies and Quantum Approximate Optimization" (https://arxiv.org/pdf/2208.05637.pdf).

## How to Use ##

Download this repository.
```
git clone https://github.com/Ericolony/quantum_optimization.git
```

Get into working directory.
```
cd quantum_optimization
```


### 1. Environment Setup ###
Follow the following instructions
```
pip install numpy
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install torch torchvision torchaudio
pip install tensorboardX
pip install backpack-for-pytorch
pip install pytorch-model-summary
pip install openfermionpsi4
pip install yacs

curl "http://vergil.chemistry.gatech.edu/psicode-download/Psi4conda-1.7-py38-Linux-x86_64.sh" -o Psi4conda-1.7-py38-Linux-x86_64.sh --keepalive-time 2
bash Psi4conda-1.7-py38-Linux-x86_64.sh -b -p $HOME/psi4conda
echo $'. $HOME/psi4conda/etc/profile.d/conda.sh\nconda activate vmc_fnc' >> ~/.bashrc
echo "source $HOME/psi4conda/etc/profile.d/conda.csh\nconda activate vmc_fnc" >> ~/.tcshrc
psi4 --test
```


### 2. Demo ###

Run the following script for evaluations of the maxcut algorithms on a graph instance with 20 nodes

```
./run.sh
```

the results are saved in

```
./logger
```
