import numpy as np
cimport numpy as cnp
cimport cython
cimport libc.math
from libcpp cimport bool

cdef extern from "zspmv.hpp" nogil:
    void zspmvpy(double complex *data, int *ind, int *ptr, double complex *vec,
                double complex a, double complex *out, int nrows)
                
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] cy_ode_rhs(
        double t,
        complex[::1] rho,
        complex[::1] data,
        int[::1] ind,
        int[::1] ptr):

    cdef unsigned int nrows = rho.shape[0]
    cdef cnp.ndarray[complex, ndim=1, mode="c"] out = \
        np.zeros(nrows, dtype=complex)
    zspmvpy(&data[0], &ind[0], &ptr[0], &rho[0], 1.0, &out[0], nrows)

    return out