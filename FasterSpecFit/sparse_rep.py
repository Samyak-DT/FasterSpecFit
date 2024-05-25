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
    def rmatvec_fast(starts, ends, values, v):
        
        nvars = len(starts)
        
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
        return self.rmatvec_fast(self.starts,
                                 self.ends,
                                 self.values,
                                 v) # only ever called with 1D arrays
    
    @staticmethod
    @jit(nopython=True, fastmath=True, nogil=True)
    def matvec_fast(starts, ends, values, nbins, v):
        
        nvars = len(starts)
        
        p = np.zeros(nbins)
        for i in range(nvars):
            vals = values[i]    # column i
            for j in range(ends[i] - starts[i]):
                p[j + starts[i]] += vals[j] * v[i]  

        return p

    # Compute matrix-vector product A * v, where A is us
    def _matvec(self, v):
        return self.matvec_fast(self.starts,
                                self.ends,
                                self.values,
                                self.shape[0], # nbins
                                v) # only ever called with 1D arrays w/matmat
    
  
    @staticmethod
    @jit(nopython=True, fastmath=True, nogil=True)
    def matmat_fast(starts, ends, values, nbins, X):

        nvars = len(starts)
        nout = X.shape[1]
        
        p = np.zeros((nbins, nout), dtype=X.dtype)
        
        for k in range(nvars):
            vals = values[k]    # column k
            for i in range(ends[k] - starts[k]):
                v = vals[i]
                for j in range(nout):
                    p[i + starts[k], j] += v * X[k,j]  
        
        return p

    # Compute matrix-matrixproduct A * X, where A is us
    def _matmat(self, X):
        return self.matmat_fast(self.starts,
                                self.ends,
                                self.values,
                                self.shape[0],
                                X)
