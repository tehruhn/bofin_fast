# distutils: language = c++
# distutils: sources = bofincpp/utilities.cpp

import numpy as np
import math
cimport numpy as cnp
cimport cython
cimport libc.math
from libcpp.map cimport map as cpp_map
from libcpp.vector cimport vector as cpp_vector
from libcpp.utility cimport pair as cpp_pair
from scipy.sparse import *

ctypedef cpp_pair[int, int] element
ctypedef cpp_map[element, double complex] custom_map
ctypedef cpp_pair[custom_map, int] custom_pair

# Imports the functions from C++

cdef extern from "utilities.h" nogil:
     custom_pair boson_py_interface(double complex* H1, int szh, double complex* C1, int szc, int szc1,
     double complex* ck1, int szck, double complex* vk1, int szvk, int nc, int nr, int ni, int k)

cdef extern from "utilities.h" nogil:
    custom_pair fermion_py_interface(double complex* H1, int szh, double complex* C1, int szc1, int szc, 
    double complex* flat_ck1, int szck, double complex* flat_vk1, int szvk, int* len_list, int sz, int nc, int k)

# Interfaces with the bosonic part of C++

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef boson_interfacecpp(
        complex[::1] H1,
        int szh,
        complex[::1] C1,
        int szc,
        int szc1,
        complex[::1] ck1,
        int szck,
        complex[::1] vk1,
        int szvk,
        int nc,
        int nr,
        int ni,
        int k):

    cdef custom_pair output
    output = boson_py_interface(&H1[0], szh, &C1[0], szc, szc1, &ck1[0], szck, &vk1[0], szvk, nc, nr, ni, k)

    if k == 2:
        szh = int(math.sqrt(szh))
    RHS = dok_matrix((szh*szh*output.second, szh*szh*output.second), dtype=complex)

    for element in output.first:
        RHS[element.first.first, element.first.second] = element.second
    RHS = RHS.tocsr()
    return RHS, output.second

# Interfaces with the fermionic part of C++

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef fermion_interfacecpp(
        complex[::1] H1,
        int szh,
        complex[::1] C1,
        int szc,
        int szc1,
        complex[::1] flat_ck1,
        int szck,
        complex[::1] flat_vk1,
        int szvk,
        int[::1] len_list,
        int sz,
        int nc,
        int k):
    cdef custom_pair output

    output = fermion_py_interface(&H1[0], szh, &C1[0], szc, szc1, &flat_ck1[0], szck, &flat_vk1[0], szvk, &len_list[0], sz, nc, k)

    if k == 2:
        szh = int(math.sqrt(szh))
    RHS = dok_matrix((szh*szh*output.second, szh*szh*output.second), dtype=complex)

    for element in output.first:
        RHS[element.first.first, element.first.second] = element.second
    RHS = RHS.tocsr()
    return RHS, output.second

