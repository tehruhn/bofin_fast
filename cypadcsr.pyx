import numpy as np
import scipy.sparse as sp

def _pad_csr(object mat, int row_scale, int col_scale, int insertrow=0, 
	int insertcol=0):
    """
    Expand the input csr_matrix to a greater space as given by the scale.
    Effectively inserting A into a larger matrix
         zeros([A.shape[0]*row_scale, A.shape[1]*col_scale]
    at the position [A.shape[0]*insertrow, A.shape[1]*insertcol]
    The same could be achieved through using a kron with a matrix with
    one element set to 1. However, this is more efficient
    """

    # ajgpitch 2016-03-08:
    # Clearly this is a very simple operation in dense matrices
    # It seems strange that there is nothing equivalent in sparse however,
    # after much searching most threads suggest directly addressing
    # the underlying arrays, as done here.
    # This certainly proved more efficient than other methods such as stacking
    #TODO: Perhaps cythonize and move to spmatfuncs
    B = mat.copy()
    if not isinstance(B, sp.csr_matrix):
        raise TypeError("First parameter must be a csr matrix")
    cdef int nrowin = B.shape[0]
    cdef int ncolin = B.shape[1]
    cdef int nrowout = nrowin*row_scale
    cdef int ncolout = ncolin*col_scale

    B._shape = (nrowout, ncolout)
    if insertcol == 0:
        pass
    elif insertcol > 0 and insertcol < col_scale:
        B.indices = B.indices + insertcol*ncolin
        # print(type(B.indices))
    else:
        raise ValueError("insertcol must be >= 0 and < col_scale")

    if insertrow == 0:
        B.indptr = np.concatenate((B.indptr,
                        np.array([B.indptr[-1]]*(row_scale-1)*nrowin)))
    elif insertrow == row_scale-1:
        B.indptr = np.concatenate((np.array([0]*(row_scale - 1)*nrowin),
                                   B.indptr))
    elif insertrow > 0 and insertrow < row_scale - 1:
         B.indptr = np.concatenate((np.array([0]*insertrow*nrowin), B.indptr,
                np.array([B.indptr[-1]]*(row_scale - insertrow - 1)*nrowin)))
    else:
        raise ValueError("insertrow must be >= 0 and < row_scale")
    # print("here3")
    return B