# Helper functions 

from numpy import kron, array, ndarray, exp, cos, sin, radians, degrees, sqrt
from qiskit import QuantumCircuit

from random import randint
from collections import namedtuple

import itertools

def flatten(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))

def tens(*ops) -> ndarray:
    """
    Recursive tensor operation. Used because numpy.kron
    only takes in two arguments.
    :param ops: List of vectors or matrices to tensor.
    :return: The tensor product of the arguments.
    """
    num_ops = len(ops)
    if num_ops == 0:
        raise ValueError("Need at least one argument for tensor product")

    if num_ops == 1:
        return ops[0]

    if num_ops == 2:
        return kron(ops[0], ops[1])

    return tens(ops[0], tens(*ops[1:]))


def gen_qubit(theta: float, phi: float) -> ndarray:
    """
    Generate a pure-stat qubit on the bloch sphere.
    Returns alpha = cos(theta/2), beta = e^(j*phi)*sin(theta/2)
    :param theta: Latitude angle.
    :param phi: Longitude angle.
    :return: Qubit vector.
    """
    return array([[cos(theta / 2)], [exp(1j * phi) * sin(theta / 2)]])


def get_density_op(state_vector: ndarray) -> ndarray:
    """
    Returns the density operator from a statevector ket.
    :param state_vector: State vector.
    :return: Density operator.
    """
    return state_vector @ state_vector.T.conj()

def gen_rand_qubit():
    theta = radians(randint(-180, 180))
    phi = radians(randint(-180, 180))
    return gen_qubit(theta, phi), theta, phi


def print_info(num, *args):
    print(f'Psi {num}, theta {degrees(args[1])}°, phi {degrees(args[2])}°')


def add_basis_meas(circuit, qubit, basis='z') -> QuantumCircuit:
    circ = circuit.copy()
    circ.barrier()

    if basis == 'x':
        circ.h(qubit)
    elif basis == 'y':
        circ.sdg(qubit)
        circ.h(qubit)

    circ.measure_all()

    return circ


# Create named tuple object, for easy indexing, e.g. ".x", instead of "[0]"
TomographySet = namedtuple('TomographySet', ['x', 'y', 'z'])


def get_tomography_circuits(circuit, qubit) -> TomographySet[QuantumCircuit]:
    return TomographySet(
        add_basis_meas(circuit, qubit, 'x'),
        add_basis_meas(circuit, qubit, 'y'),
        add_basis_meas(circuit, qubit, 'z')
    )


def bloch_oordinate(counts_dict: dict) -> float:
    if '1' not in counts_dict:
        return 1.0
    if '0' not in counts_dict:
        return -1.0
    
    plus = counts_dict['0'] if '0' in counts_dict else 0
    minus = counts_dict['1'] if '1' in counts_dict else 0
    return (plus - minus) / (plus + minus)


def density_op_from_bloch_coordinates(rx, ry, rz):
    sqr_norm = rx ** 2 + ry ** 2 + rz ** 2
    if sqr_norm > 1:
        norm = sqrt(sqr_norm)
        rx, ry, rz = rx / norm, ry / norm, rz / norm

    op = array([[1 + rz, rx - 1j * ry],
                [rx + 1j * ry, 1 - rz]])

    return 0.5 * op


def density_op_from_counts_dict(counts_x, counts_y, counts_z):
    return density_op_from_bloch_coordinates(
        rx=bloch_oordinate(counts_x),
        ry=bloch_oordinate(counts_y),
        rz=bloch_oordinate(counts_z)
    )


def add_dicts(*dicts):
    output = dicts[0].copy()
    for dict in dicts[1:]:
        for key in dict.keys():
            output[key] += dict[key]
    return output


