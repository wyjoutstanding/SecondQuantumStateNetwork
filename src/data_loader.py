import os
import numpy as np
from src.data.chemistry.read_hamiltonian import load_molecule



def load_hamiltonian_string(molecule_name, basis, data_path, hamiltonian_type):
    """
    Load the Hamiltonian string from the given data_path.
    """
    
    # _, qubit_hamiltonian = load_molecule(data_path, hamiltonian_fname=None, compute_gt=compute_gt)
    molecule, qubit_hamiltonian = load_molecule(molecule_name, basis, data_path, hamiltonian_type)
    if hamiltonian_type == 'exact':
        qubit_hamiltonian = f"{qubit_hamiltonian}"
    return molecule, qubit_hamiltonian


def load_data(cfg):
    """
    Load the data according to the given configuration (cfg).
    """
    data_path = cfg.DATA.LOAD_PATH
    molecule_name = cfg.DATA.MOLECULE
    basis = cfg.DATA.BASIS
    hamiltonian_type = cfg.HAMILTONIAN.TYPE

    # if not os.path.exists(data_path):
    #     raise ValueError("Data path does not exist!")
    molecule, string = load_hamiltonian_string(molecule_name, basis, data_path, hamiltonian_type)
    num_sites = molecule.n_qubits
    data = {'hamiltonian_type': hamiltonian_type, 'hamiltonian_string': string, 'num_sites': num_sites, 'molecule': molecule}
    return data
