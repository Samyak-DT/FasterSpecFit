import numpy as np
import scipy.sparse as sp

from numba import jit


#
# Sparse representation customized to the
# structure of the emline fitting Jacobian.
# This representation is produced directly by
# our Jacobian calculation.
#

class EMLineSparseArray(sp.linalg.LinearOperator):
    
    # 'values' is a 2D array whose ith row contains
    # the nonzero values in column i of the matrix.
    # These entries correspond to entries
    # [starts[i] .. ends[i] - 1] of the dense column.
    # (Note that not every column has the same number
    # of nonzero entries, so we need both start and end.)
    def __init__(self, shape, starts, ends, values):

        self.starts = starts
        self.ends   = ends
        self.values = values

        super().__init__(values.dtype, shape)

    @staticmethod
    @jit(nopython=True, fastmath=True, nogil=True)
    def rmatvec_fast(nbins, nvars, starts, ends, values, v):

        p = np.empty(nvars)
        for i in range(nvars):
            acc = 0.
            vals = values[i]   # row i of transpose
            
            for j in range(ends[i] - starts[i]):
                acc += vals[j] * v[j + starts[i]]
            p[i] = acc
        
        return p

    # Compute matrix-vector product A.T * v, where A is us
    def _rmatvec(self, v):

        nbins, nvars = self.shape

        return self.rmatvec_fast(nbins, nvars, \
                                 self.starts,
                                 self.ends,
                                 self.values,
                                 v.flatten())

    @staticmethod
    @jit(nopython=True, fastmath=True, nogil=True)
    def matvec_fast(nbins, nvars, starts, ends, values, v):

        p = np.zeros(nbins)
        for i in range(nvars):
            vals = values[i]    # column i
            for j in range(ends[i] - starts[i]):
                p[j + starts[i]] += vals[j] * v[i]  

        return p

    # Compute matrix-vector product A * v, where A is us
    def _matvec(self, v):
        
        nbins, nvars = self.shape
        
        return self.matvec_fast(nbins, nvars,
                                self.starts,
                                self.ends,
                                self.values,
                                v.flatten())
