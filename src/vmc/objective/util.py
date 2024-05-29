from .hamiltonian import MolecularHamiltoian, parse_hamiltonian_string
from .hamiltonian import MolecularHamiltonianCPP

def get_hamiltonian(hamiltonian_type='exact', **kwargs):
    """
    Returns an instance of Hamiltonian based on the problem type and kwargs.
    Args:
        problem_type (str): Type of problem, can be 'str_auto' or 'qc'.
        **kwargs: Additional keyword arguments for parsing the Hamiltonian string.
    Returns:
        Hamiltonian: An instance of Hamiltonian.
    """
    if hamiltonian_type == 'exact':
        matrix, coefficients = parse_hamiltonian_string(**kwargs)
        return MolecularHamiltoian(matrix, coefficients)
    else:
        return MolecularHamiltonianCPP(kwargs['hamiltonian_string'])
