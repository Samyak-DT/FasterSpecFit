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
@jit(nopython=True, fastmath=True, nogil=True)
def resMul(M, v):

    nrow, ndiag = M.shape
    hdiag = ndiag//2

    w = np.empty(nrow, dtype=v.dtype)

    for i in range(nrow):
        jmin = np.maximum(i - hdiag,    0)
        jmax = np.minimum(i + hdiag, nrow - 1)

        acc = 0.
        for j in range(jmin, jmax + 1):
            acc += M[i, j - i + hdiag] * v[j]

        w[i] = acc

    return w


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
def mulWMJ(w, M, Jtuple):

    starts, ends, J = Jtuple
    
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
        
        # boundaries of nonzero entries
        # in jth column of P
        imin = np.maximum(s - hdiag, 0)
        imax = np.minimum(e + hdiag, nbins - 1)
        
        for i in range(imin, imax):
            
            # boundaries of interval of k where both
            # M[i, k] and J[k, j] are nonzero.
            kmin = np.maximum(i - hdiag,     s)
            kmax = np.minimum(i + hdiag, e - 1)
            
            acc = 0.
            for k in range(kmin, kmax + 1):
                acc += M[i, k - i + hdiag] * J[j, k - s]
            P[j][i - imin] = acc * w[i]
            
        startsP[j] = np.maximum(imin, 0)
        endsP[j]   = np.minimum(imax, nbins - 1)
    
    return (startsP, endsP, P)


class ParamsMapping(object):

    def __init__(self, nParms,
                 freeParms, tiedParms, tiedSources, tiedFactors,
                 doubletRatios, doubletSources):

        nFree = len(freeParms)
        
        J_S = np.zeros((nParms, nFree))
        
        # permutation mapping each free parameter in full list
        # to its location in free list
        p = np.empty(nParms, dtype=np.int32)
        p[freeParms] = np.arange(nFree, dtype=np.int32)
        
        # free params present unchanged in full list
        for j in freeParms:
            J_S[j,p[j]] = 1.

        # tied params
        for j, src_j, factor in zip(tiedParms, tiedSources, tiedFactors):
            if src_j not in freeParms:
                #print(f"SOURCE {src_j} tied to {j} is not free!")
                # if source is fixed, so is target, so Jacobian contrib is 0
                pass
            else:
                J_S[j, p[src_j]] = factor
        
        # create placeholders for doublets in sparse structure
        # and record ops needed to compute Jacobian for given free params
        doubletPatches = np.empty((len(doubletRatios), 3), dtype=np.int32)
        idx = 0
        for (j, src_j) in zip(doubletRatios, doubletSources):
            #if j not in freeParms:
            #    print(f"ratio {j} in doublet with {src_j} is not free!")
            #if src_j not in freeParms:
            #    print(f"amplitude {src_j} in doublet with {j} is not free!")

            if j not in freeParms or src_j not in freeParms:
                continue
            
            J_S[j,p[src_j]] = 1. # will set to v[j]
            J_S[j,p[j]]     = 1. # will set to v[src_j]
            doubletPatches[idx] = np.array([j, p[src_j], p[j]])
            idx += 1
            
        self.J_S = sp.csr_array(J_S)
        self.doubletPatches = doubletPatches[:idx,:]
        
    # evaluate Jacobian at v = freeParms
    def getJacobian(self, freeParms):
        for j, j_free, src_j_free in self.doubletPatches:
            self.J_S[j, j_free]     = freeParms[src_j_free]
            self.J_S[j, src_j_free] = freeParms[j_free]
        
        return self.J_S

    
class EMLineJacobian(sp.linalg.LinearOperator):

    # we will pass camerapix, jacs, and J_S.  Jacs will be outputs of mulWMJ.
    # Move JIT'd multiplies from IdealJacobian into here to avoid another
    # layer of indirection.
    
    def __init__(self, camerapix, jacs, J_S):

        nFreeParms = J_S.shape[1]
        
        nBins = 0
        for campix in camerapix:
            nBins += campix[1] - campix[0]
        
        self.camerapix = camerapix
        self.jacs      = jacs
        self.J_S       = J_S
        
        super().__init__(J_S.dtype, (nBins, nFreeParms))
        

    @staticmethod
    @jit(nopython=True, fastmath=True, nogil=True)
    def matvec_fast(J, v, nbins):

        starts, ends, values = J
        nvars = len(starts)
        
        p = np.zeros(nbins)
        for i in range(nvars):
            vals = values[i]    # column i
            for j in range(ends[i] - starts[i]):
                p[j + starts[i]] += vals[j] * v[i]  

        return p

    # |v| = number of free parameters
    def _matvec(self, v):
        
        vFull = self.J_S.dot(v.ravel())
        
        # everything below here should be JIT'd. resM and final weight mul go away
        nBins = self.shape[0]
        w = np.empty(nBins, dtype=vFull.dtype)
        
        for campix, jac, in zip(self.camerapix, self.jacs):
            s = campix[0]
            e = campix[1]
            
            w[s:e] = self.matvec_fast(jac, vFull, e - s)
            
        return w

    @staticmethod
    @jit(nopython=True, fastmath=True, nogil=True)
    def rmatvec_fast(J, v):

        starts, ends, values = J
        nvars = len(starts)
        
        p = np.empty(nvars)
        for i in range(nvars):
            acc = 0.
            vals = values[i]   # row i of transpose
            
            for j in range(ends[i] - starts[i]):
                acc += vals[j] * v[j + starts[i]]
            p[i] = acc
        
        return p

    # |v| = number of observable bins
    def _rmatvec(self, v):
        
        nFreeParms = self.shape[1]    
        w = np.zeros(nFreeParms, dtype=v.dtype)
        
        # mulWMJ should produce a replacement for each ideal Jacobian that includes
        # the obs weights, the resm, and the ideal Jacobian
        for campix, jac in zip(self.camerapix, self.jacs):
            s = campix[0]
            e = campix[1]

            vSub = v[s:e]
            vJac = self.rmatvec_fast(jac, vSub)

            # this has to be Numbafied before we can JIT the loop
            vFreeParms = self.J_S.T.dot(vJac) 
            w += vFreeParms

        return w


    
"""    
#
# Sparse representation customized to the
# structure of the emline fitting ideal Jacobian.
# This representation is produced directly by
# our ideal Jacobian calculation.
#

class EMLineIdealJacobian(sp.linalg.LinearOperator):
    
    # 'values' is a 2D array whose ith row contains
    # the nonzero values in column i of the matrix.
    # These entries correspond to entries
    # [starts[i] .. ends[i] - 1] of the dense column.
    # (Note that not every column has the same number
    # of nonzero entries, so we need both start and end.)
    def __init__(self, shape, Jtuple):

        starts, ends, values = Jtuple
        
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
"""
