from math import sqrt

import numpy
from numpy import array, identity
from allens_quantum_package.functions import *

# Constants
invSqrtTwo = 1 / sqrt(2)

# 1-qubit standard basis vectors
zro = array([[1], [0]])
one = array([[0], [1]])

# 1-qubit operators
I = identity(2)
X = array([[0, 1], [1, 0]])
Y = array([[0, -1j], [1j, 0]])
Z = array([[1, 0], [0, -1]])
H = invSqrtTwo * array([[1, 1], [1, -1]])

# 1-qubit Hadamard basis vectors
pls = H @ zro
mns = H @ one

# 2-qubit standard basis vectors
zrozro = tens(zro, zro)
zroone = tens(zro, one)
onezro = tens(one, zro)
oneone = tens(one, one)

# 2-qubit operators
CNOT = zrozro @ zrozro.T + zroone @ zroone.T + \
       onezro @ oneone.T + oneone @ onezro.T

# 3-qubit standard vectors
zrozrozro = tens(zro, zro, zro)
zrozroone = tens(zro, zro, one)
zroonezro = tens(zro, one, zro)
zrooneone = tens(zro, one, one)
onezrozro = tens(one, zro, zro)
onezroone = tens(one, zro, one)
oneonezro = tens(one, one, zro)
oneoneone = tens(one, one, one)


def gate_builder(*pairs: tuple[ndarray, ndarray]) -> ndarray:
    '''
    Creates a gate operator by taking an iterable of tuples. Returns the
    sum of input @ output.T iterables.
    :param tuples: Tuple in the form of (input, output)
    :return: Operator array.
    '''
    return numpy.sum(tuple[0] @ tuple[1].T.conj() for tuple in pairs)

# 3-qubit operators
CSWAP = gate_builder((zrozrozro, zrozrozro), (zrozroone, zrozroone),
                     (zroonezro, zroonezro), (zrooneone, onezroone),
                     (onezrozro, onezrozro), (onezroone, zrooneone),
                     (oneonezro, oneonezro), (oneoneone, oneoneone))

CSWAP_01 = CSWAP

CSWAP_10 = gate_builder((zrozrozro, zrozrozro), (zrozroone, zrozroone),
                        (zroonezro, zroonezro), (zrooneone, oneonezro),
                        (onezrozro, onezrozro), (onezroone, onezroone),
                        (oneonezro, zrooneone), (oneoneone, oneoneone))

CSWAP_02 = gate_builder((zrozrozro, zrozrozro), (zrozroone, zrozroone),
                        (zroonezro, zroonezro), (zrooneone, onezroone),
                        (onezrozro, onezrozro), (onezroone, zrooneone),
                        (oneonezro, oneonezro), (oneoneone, oneoneone))

CSWAP_20 = gate_builder((zrozrozro, zrozrozro), (zrozroone, zrozroone),
                        (zroonezro, zroonezro), (zrooneone, zrooneone),
                        (onezrozro, onezrozro), (onezroone, oneonezro),
                        (oneonezro, onezroone), (oneoneone, oneoneone))
