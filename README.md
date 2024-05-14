# Second Quantum States Network Framework for Quantum Chemistry
General second quantum states network framework for quantum chemistry.
Support Variational Monte Carlo optimizations:
- Sampling: Batch Autoregressive Sampling
- Local energy calculation: based on Pauli strings, support CPU and GPU
- Gradient: support by PyTorch
- Optimizer: Stochastic Reconfiguration and gradient descent optimizer(e.g. AdamW, SGD...)
- Wavefunctions: support NADE and MADE now

## How to Use

### 1. Environment Setup

To set up the environment, follow these instructions:
#### Install Psi4 conda environment (strongly recommand)
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

## Developments

### Adding a custom wavefunction
To implement a new wavefunction class by inheriting from `Base` and implementing the required `forward` and `sample` methods, follow these steps:

#### Step-by-Step Guide:

1. **Create a New Class**:
   - Inherit from the `src.models.base.Base` class.
   - Implement the `forward` and `sample` interfaces.

2. **Implement the `forward` Interface**:
   - This method should compute the logarithm of the wavefunction for given configurations.
```python
def forward(self, configurations):
    r"""
    Args:
        configurations: Tensor of shape [batch_size, n_qubits]
                        Each element is either +1 or -1, where +1 indicates an occupied qubit.
    Returns: 
        ln_psi: Tensor of shape [batch_size, 2], representing the natural logarithm of the (ln|ψ and phase). The output is in a complex format, where the second dimension represents the ln(amplitude) and phase parts.
    
    Example:
        configurations = torch.tensor([[1, -1, 1, -1], [-1, 1, -1, 1]])
        ln_psi = model.forward(configurations)
    """
```

3. **Implement the `sample` Interface**:
   - This method should generate sample configurations based on the specified batch size.
```python
def sample(self, batch_size=1000):
    r"""
    Args:
        batch_size: the number of samples to generate. Default is 1e12.
    
    Returns:
        configurations: Tensor of shape [:, n_qubits], representing the sampled configurations.
                        Each row is a sampled configuration where each element is either +1 or -1.
    
    Example:
        model.set_sampling(True)
        sampled_configs = model.sample(batch_size=1000)
    """
```

#### Example Implementation:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.models.base import Base  # Ensure Base is imported correctly from your module

class MyWaveFunction(Base):
    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
        self.fc1 = nn.Linear(n_qubits, 128)
        self.fc2 = nn.Linear(128, 2)  # Output size 2 for real and imaginary parts

    def forward(self, configurations):
        x = self.fc1(configurations.float())
        x = F.relu(x)
        ln_psi = self.fc2(x)
        return ln_psi

    def sample(self, batch_size=1000):
        configurations = torch.randint(0, 2, (batch_size, self.n_qubits)) * 2 - 1
        return configurations

# Example usage
if __name__ == "__main__":
    n_qubits = 4
    model = MyWaveFunction(n_qubits)
    
    # Generate sample configurations
    sample_configs = model.sample(batch_size=10)
    print("Sampled Configurations:\n", sample_configs)
    
    # Forward pass through the model
    ln_psi = model.forward(sample_configs)
    print("Log Psi:\n", ln_psi)
```

#### Example Explanation:

1. **Class Definition**:
   - `MyWaveFunction` inherits from `Base`.
   - Initialize with the number of qubits and define the neural network layers.

2. **Forward Method**:
   - Takes configurations of shape `[batch_size, n_qubits]`.
   - Passes through two fully connected layers with ReLU activation.
   - Outputs `ln_psi` with shape `[batch_size, 2]`.

3. **Sample Method**:
   - Generates random configurations of +1 or -1 with specified batch size.

This implementation provides a simple example of how to extend the `Base` class and implement the required methods for a specific wavefunction. More practical example can see the `nade.py` or `made.py`.