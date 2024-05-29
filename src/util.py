
import os
import random
import logging
import numpy as np
import torch
from datetime import datetime

import numpy
import openfermion

def save_binary_qubit_op(op: openfermion.QubitOperator,
                         filename: str = "qubit_op.data"):
    """
    Convert the op into a binay file representation.
    The file structure is:
    float: 0x4026828f5c28f5c3 (11.2552), an identifier,
    int32: Number of qubits,
    double, double: Real and imaginary part of the coefficient,
    int32, int32, ...: X/Y/XI, X/Y/Z/I, ... (repeat n_qubits times).

    Args:
        op (QubitOperator): The qubit operator to be stored.
        filename (str): The name of the file.

    Returns:
        size_file (int): The size of the file in bytes.

    Notes:
        All ints are int32, double is float64.
    """
    if type(op) is not openfermion.QubitOperator:
        raise TypeError("op must be a QubitOperator but got {}.\
".format(type(op)))
    if type(filename) is not str:
        raise TypeError("filename must be a string but got {}.\
".format(type(filename)))

    n_qubits = openfermion.count_qubits(op)
    size_file = 0

    f = open(filename, "wb")
    n_qubits_array = numpy.array([n_qubits], dtype=numpy.int32)
    coeffs_array = numpy.zeros([2], dtype=numpy.float64)
    pauli_str_array = numpy.zeros([n_qubits], dtype=numpy.int32)
    get_pauli_number = {
        "I": 0,
        "X": 1,
        "Y": 2,
        "Z": 3
    }

    # f.write(bytes.fromhex("4026828f5c28f5c3"))  # 0x402682a9930be0df
    f.write(numpy.array([11.2552], dtype=numpy.float64).tobytes())
    size_file += 8

    f.write(n_qubits_array.tobytes())
    size_file += 4
    for pauli_term in op.terms:
        coeff = complex(op.terms[pauli_term])
        coeffs_array[0] = coeff.real
        coeffs_array[1] = coeff.imag
        f.write(coeffs_array.tobytes())
        size_file += 16
        coeffs_array.fill(0)

        for pos, pauli_symbol in pauli_term:
            pauli_str_array[pos] = get_pauli_number[pauli_symbol]
        f.write(pauli_str_array.tobytes())
        size_file += n_qubits * 4
        pauli_str_array.fill(0)

    f.close()

    return size_file

def read_binary_qubit_op(filename: str = "qubit_op.data"):
    """
    """
    f = open(filename, "rb")
    identifier = f.read(8)
    # if identifier != bytes.fromhex("4026828f5c28f5c3"):  # 0x402682a9930be0df
    if numpy.frombuffer(identifier, dtype=numpy.float64) != 11.2552:
        raise ValueError("The file is not saved by QCQC.")

    n_qubits = numpy.frombuffer(f.read(4), dtype=numpy.int32)
    n_qubits = int(n_qubits)

    get_pauli_symbol = {
        0: "I",
        1: "X",
        2: "Y",
        3: "Z"
    }

    qubit_op = openfermion.QubitOperator()

    pauli_str_array = numpy.zeros([n_qubits], dtype=numpy.int32)
    chunk_size = n_qubits * 4
    coeff_bin = f.read(16)
    pauli_str_bin = f.read(chunk_size)
    while len(coeff_bin) != 0 and len(pauli_str_bin) != 0:
        assert(len(pauli_str_bin) == chunk_size)
        coeffs_array = numpy.frombuffer(coeff_bin, dtype=numpy.float64)
        pauli_str_array = numpy.frombuffer(pauli_str_bin, dtype=numpy.int32)
        coeff = coeffs_array[0] + 1.j * coeffs_array[1]
        term_list = []
        for pos, pauli_number in enumerate(pauli_str_array):
            pauli_symbol = get_pauli_symbol[pauli_number]
            if pauli_symbol != "I":
                term_list.append((pos, pauli_symbol))
        op_cur = openfermion.QubitOperator(tuple(term_list), coeff)
        qubit_op += op_cur

        coeff_bin = f.read(16)
        pauli_str_bin = f.read(chunk_size)

    return qubit_op

def set_seed(seed):
    # the followings are for reproducibility on GPU, see https://pytorch.org/docs/master/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")

def folder_name_generator(cfg, opts):
    name_str = []
    name_str.append('{}'.format(cfg.MISC.MODE))
    for i,arg in enumerate(opts):
        if i % 2 == 1:
            if opts[i-1] == 'DATA.LOAD_PATH':
                continue
            name_str.append('{}'.format(arg))
    return '-'.join(name_str)

def prepare_dirs(cfg):
    if not os.path.exists('./datasets'):
        os.makedirs('./datasets')
    if not os.path.exists('./results'):
        os.makedirs('./results')
    if not os.path.exists('./logger'):
        os.makedirs('./logger')
    if not os.path.exists(cfg.MISC.DIR):
        os.makedirs(cfg.MISC.DIR)
    # set logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(cfg.MISC.DIR, 'debug.log')),
            logging.StreamHandler()
        ]
    )

def write_file(file_name, content, local_rank=0):
    # only write to disk on the master thread
    if local_rank == 0:
        f=open(file_name, "a+")
        f.write(content)
        f.write("\n")
        f.close()

