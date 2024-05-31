import numpy as np
import scipy.sparse as sp

from numba import jit

#
# dia_to_row_matrix()
# Convert a diagonally sparse matrix M in the form
# stored by DESI into a sparse row rerpesentation.
#
# Input M is represented as a 2D array D of size ndiag x nrow,
# whose rows are M's diagonals:
#            M[i,j] = D[ndiag//2 - (j - i), j]
# ndiag is assumed to be odd, and entries in D that would be
# outside the bounds of M are ignored.
#
# Output M is a 2D array A of size nrow x ndiag,,whose rows are
# M's rows, but with only the nonzero entries stored.  The nonzero
# entries on row i run for j = i - diag//2 to i + diag//2, so
#            M[i,j] = A[i, j - (i - diag//2)]
#
@jit(nopython=True, fastmath=True, nogil=True)
def dia_to_row_matrix(D):

    ndiag, nrow = D.shape
    hdiag = ndiag//2

    A = np.empty((nrow, ndiag), dtype=D.dtype)
    
    for i in range(nrow):
        # min and max column for row
        jmin = np.maximum(i - hdiag,        0)
        jmax = np.minimum(i + hdiag, nrow - 1)
        for j in range(jmin, jmax + 1):
            A[i, j - i + hdiag] = D[hdiag + i - j, j]
        
    return A


#
# resMul()
# Compute the matrix-vector product Mv, where
# M is a row-sparse matrix with a limited
# number of diagonals created by
# dia_to_row_matrix().
#
# w is an output parameter
@jit(nopython=True, fastmath=True, nogil=True)
def resMul(M, v, w):

    nrow, ndiag = M.shape
    hdiag = ndiag//2

    for i in range(nrow):
        jmin = np.maximum(i - hdiag,    0)
        jmax = np.minimum(i + hdiag, nrow - 1)

        acc = 0.
        for j in range(jmin, jmax + 1):
            acc += M[i, j - i + hdiag] * v[j]
        
        w[i] = acc

#
# mulWMJ()
# Compute the product WMJ, where
#   W is a diagonal matrix (represented by a vector)
#   M is a row-sparse matrix computed by dia_to_row()
#   J is a column-sparse matrix computed by the ideal
#     Jacobian calculation.
#
# Return the result as a column-sparse matrix in the
# same form as J.
#
# NB: for future, we shoudl allocate enough extra space
# in J to let us create P in place, overwriting J
#
@jit(nopython=True, fastmath=True, nogil=True)
def mulWMJ(w, M, jac):

    starts, ends, J = jac
    
    nbins, ndiag = M.shape
    ncol, maxColSize = J.shape
    
    hdiag = ndiag//2
    
    P = np.empty((ncol, maxColSize + ndiag - 1), dtype=J.dtype)
    startsP = np.zeros(ncol, dtype=np.int32)
    endsP   = np.zeros(ncol, dtype=np.int32)
    
    for j in range(ncol):
        # boundaries of nonzero entries
        # in jth column of J
        s = starts[j]
        e = ends[j]
        
        if s == e: # no nonzero values in column j
            continue
        
        # boundaries of entries in jth column of P
        # impacted by matrix multiply
        imin = np.maximum(s - hdiag, 0)
        imax = np.minimum(e + hdiag, nbins) # one past last impacted entry
        
        for i in range(imin, imax):
            
            # boundaries of interval of k where both
            # M[i, k] and J[k, j] are nonzero.
            kmin = np.maximum(i - hdiag,     s)
            kmax = np.minimum(i + hdiag, e - 1)
        
            acc = 0.
            for k in range(kmin, kmax + 1):
                acc += M[i, k - i + hdiag] * J[j, k - s]
            P[j, i - imin] = acc * w[i]
        
        startsP[j] = np.maximum(imin, 0)
        endsP[j]   = np.minimum(imax, nbins)
                
    return (startsP, endsP, P)



class EMLineJacobian(sp.linalg.LinearOperator):

    # we will pass camerapix, jacs, and J_S.  Jacs will be outputs of mulWMJ.
    # Move JIT'd multiplies from IdealJacobian into here to avoid another
    # layer of indirection.
    
    def __init__(self, shape, camerapix, jacs, J_S):
        
        self.camerapix = camerapix
        self.jacs      = jacs
        self.J_S       = J_S

        # temporary storage for intermediate result
        nParms = jacs[0][2].shape[0]
        self.vFull = np.empty(nParms, dtype=J_S[2].dtype)
                
        super().__init__(J_S[2].dtype, shape)


    # |v| = number of free parameters
    def _matvec(self, v):

        nBins = self.shape[0]
        w = np.zeros(nBins, dtype=v.dtype)

        self._matvec_JS(self.J_S, v.ravel(), self.vFull)
        
        for campix, jac in zip(self.camerapix, self.jacs):
            s = campix[0]
            e = campix[1]
            
            self._matvec_J(jac, self.vFull, w[s:e])
            
        return w
        
    
    # |v| = number of observable bins
    def _rmatvec(self, v):

        nFreeParms = self.shape[1]
        w = np.zeros(nFreeParms, dtype=v.dtype)
        
        for campix, jac in zip(self.camerapix, self.jacs):
            s = campix[0]
            e = campix[1]
            
            self._rmatvec_J(jac, v[s:e], self.vFull)

            self._rmatvec_JS(self.J_S, self.vFull, w)
            
        return w

    @staticmethod
    @jit(nopython=True, fastmath=True, nogil=True)
    def _matvec_JS(J_S, v, w):
    
        shape, ops, factors = J_S
        
        for j in range(len(w)):
            w[j] = 0.
        
        for i in range(len(ops)):
            w[ops[i, 0]] += factors[i] * v[ops[i, 1]]
        return w


    # avoid allocations by passing in destination vec w
    @staticmethod
    @jit(nopython=True, fastmath=True, nogil=True)
    def _rmatvec_JS(J_S, v, w):

        shape, ops, factors = J_S
    
        for i in range(len(ops)):
            w[ops[i, 1]] += factors[i] * v[ops[i, 0]]


    # avoid allocations by passing in destination vec w
    @staticmethod
    @jit(nopython=True, fastmath=True, nogil=True)
    def _matvec_J(J, v, w):
    
        starts, ends, values = J
        nvars = len(starts)
        
        for i in range(nvars):
            vals = values[i]    # column i
            for j in range(ends[i] - starts[i]):
                w[j + starts[i]] += vals[j] * v[i]  
    

    @staticmethod
    @jit(nopython=True, fastmath=True, nogil=True)
    def _rmatvec_J(J, v, w):
    
        starts, ends, values = J
        nvars = len(starts)
    
        for i in range(nvars):
            vals = values[i]   # row i of transpose
            
            acc = 0.
            for j in range(ends[i] - starts[i]):
                acc += vals[j] * v[j + starts[i]]
            w[i] = acc
            
        return w
