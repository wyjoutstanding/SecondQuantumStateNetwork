# Second Quantum States Network Framework for Quantum Chemistry
General second quantum states network framework for quantum chemistry.
Support Variational Monte Carlo optimizations:
- Sampling: Batch Autoregressive Sampling
- Local energy calculation: based on Pauli strings, support CPU and GPU
- gradient: support by PyTorch
- Optimizer: Stochastic Reconfiguration and gradient descent optimizer(e.g. AdamW, SGD...)
  
## How to Use

### 1. Environment Setup

To set up the environment, follow these instructions:
#### Install Psi4 conda environment
- Download and install Psi4 for WSL example:
    ```shell
    #download via button above  -OR-  following line
    curl "http://vergil.chemistry.gatech.edu/psicode-download/Psi4conda-1.9.1-py38-WindowsWSL-x86_64.sh" -o Psi4conda-1.9.1-py38-WindowsWSL-x86_64.sh --keepalive-time 2
    bash Psi4conda-1.9.1-py38-WindowsWSL-x86_64.sh -b -p $HOME/psi4conda
    echo $'. $HOME/psi4conda/etc/profile.d/conda.sh\nconda activate' >> ~/.bashrc
    echo "source $HOME/psi4conda/etc/profile.d/conda.csh\nconda activate" >> ~/.tcshrc
    # log out, log back in so conda and psi4 in path
    psi4 --test
    ```
- Download and install Psi4 for general linux example:

    ```shell
    curl "http://vergil.chemistry.gatech.edu/psicode-download/Psi4conda-1.7-py38-Linux-x86_64.sh" -o Psi4conda-1.7-py38-Linux-x86_64.sh --keepalive-time 2
    bash Psi4conda-1.7-py38-Linux-x86_64.sh -b -p $HOME/psi4conda
    echo $'. $HOME/psi4conda/etc/profile.d/conda.sh\nconda activate' >> ~/.bashrc
    echo "source $HOME/psi4conda/etc/profile.d/conda.csh\nconda activate" >> ~/.tcshrc
    ```

- Verify that Psi4 is working correctly:

    ```
    psi4 --test
    ```
#### Install python packages based on Psi4 conda **environment**
- Install the required packages using pip and conda:

    ```shell
    pip install -r requirements.txt
    # backpack_for_pytorch==1.6.0
    # numpy==1.24.4
    # openfermion==1.6.1
    # openfermionpsi4==0.5
    # PubChemPy==1.0.4
    # pytorch_model_summary==0.1.2
    # scipy==1.13.0
    # tensorboardX==2.6.2.2
    # torch==1.12.0
    # tqdm==4.66.1
    # yacs==0.1.8
    ```
<!-- pip install tensorboardX backpack-for-pytorch pytorch-model-summary openfermionpsi4  yacs torch -->

### 2. Demo

To run the demo, execute the following script:

```shell
./run.sh
```

The script will evaluate the H2 molecule on various basis sets and save the results in the `./logger` directory. The results include the energy and convergence information for each basis set, as well as the time taken for each calculation. Please note that the demo may take some time to run, depending on your hardware, the size of the basis sets, number of training iterations (learning rate), batch size (for larger molecules).
