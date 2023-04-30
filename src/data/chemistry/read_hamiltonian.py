import os
import time
import logging
import pickle
import pubchempy
import openfermionpsi4 as ofpsi4
from openfermion.chem import MolecularData
from openfermion.transforms import get_fermion_operator, jordan_wigner


BASIS_LIST = [
    'STO-3G',
    '3-21G',
    '6-31G',
    '6-311G*',
    '6-311+G*',
    '6-311++G**',
    '6-311++G(2df,2pd)',
    # '6-311++G(3df,3pd)',
    # 'cc-pVTZ',
    # 'cc-pVQZ',
    # 'aug-cc-pCVQZ',
]


MOLECULE_CID = {
    'H2': 783,
    "H2O": 962,
    "O2": 977,
    "CO2": 280,
    "CH4": 297,
    # "CNa2O3": 10340,
    # 'Aspirin': 2244,
    # 'VitaminC': 54670067,
}


def get_molecule_geometry(name):
    id = MOLECULE_CID[name]
    pubchempy_molecule = pubchempy.get_compounds(id, 'cid', record_type='3d')
    if len(pubchempy_molecule) == 0:
        pubchempy_molecule = pubchempy.get_compounds(id, 'cid', record_type='2d')
    pubchempy_geometry = pubchempy_molecule[0].to_dict(properties=['atoms'])['atoms']
    geometry = [(atom['element'], (atom['x'], atom['y'], atom.get('z', 0))) for atom in pubchempy_geometry]
    return geometry


def load_molecule(molecule_name, basis, data_path):
    multiplicity = 1 if molecule_name not in ["O2","CH2"] else 3
    geometry = get_molecule_geometry(molecule_name)
    save_name = './datasets/chemistry/{}/{}'.format(basis, molecule_name)
    if not os.path.exists(save_name):
        os.makedirs(save_name)
    molecule = MolecularData(geometry,
                             basis=basis,                      
                             multiplicity=multiplicity,
                             charge=0,
                             filename=save_name,
                             )
    # 1. Solve molecule and print results.
    logging.info("Solving molecule with psi4...")
    t_start=time.time()
    molecule = ofpsi4.run_psi4(molecule, memory=16000, delete_input=True, delete_output=True, verbose=True)
    logging.info(f'{molecule_name} has:')
    logging.info(f'\t\t\tbasis {basis}')
    logging.info(f'\t\t\tgeometry {molecule.geometry},')
    logging.info(f'\t\t\t{molecule.n_electrons} electrons in {2*molecule.n_orbitals} spin-orbitals,')
    logging.info(f'\t\t\tHartree-Fock energy of {molecule.hf_energy:.6f} Hartree,')
    logging.info("done in {:.2f} seconds".format(time.time()-t_start))
    # 2. Save molecule.
    molecule.save()
    logging.info(f"Molecule saved to {save_name}.hdf.")
    # 3. Convert molecular Hamiltonian to qubit Hamiltonian.
    logging.info("Converting molecular Hamiltonian to qubit Hamiltonian... at {:.2f} seconds".format(time.time()-t_start))
    active_space_start=0
    active_space_stop=molecule.n_orbitals
    logging.info("Number of qubits {}".format(molecule.n_qubits))
    logging.info("Computing the hamiltonian... at {:.2f} seconds".format(time.time()-t_start))
    # Get the Hamiltonian in an active space.
    molecular_hamiltonian = molecule.get_molecular_hamiltonian(
        occupied_indices=None,
        active_indices=range(active_space_start, active_space_stop))
    logging.info("Get Fermion Operator... at {:.2f} seconds".format(time.time()-t_start))
    fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
    logging.info("Perform Jordan Wigner... at {:.2f} seconds".format(time.time()-t_start))
    qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
    logging.info("Compressing... at {:.2f} seconds".format(time.time()-t_start))
    qubit_hamiltonian.compress()
    logging.info("done in {:.2f} seconds".format(time.time()-t_start))
    return molecule, qubit_hamiltonian

