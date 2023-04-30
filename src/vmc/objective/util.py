from .automatic import Automatic, parse_hamiltonian_string

def get_hamiltonian(**kwargs):
    """
    Returns an instance of Hamiltonian based on the problem type and kwargs.
    Args:
        problem_type (str): Type of problem, can be 'str_auto' or 'qc'.
        **kwargs: Additional keyword arguments for parsing the Hamiltonian string.
    Returns:
        Hamiltonian: An instance of Hamiltonian.
    """
    matrix, coefficients = parse_hamiltonian_string(**kwargs)
    return Automatic(matrix, coefficients)
