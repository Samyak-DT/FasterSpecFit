import numpy as np
import scipy.sparse as sp

from numba import jit


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

    
class EMLineJacobian(sp.linalg.LinearOperator):

    def __init__(self, camerapix, obs_weights, resMatrices, idealJacs, J_S):

        nFreeParms = J_S.shape[1]
        
        nBins      = 0
        for resM in resMatrices:
            nBins += resM.shape[0]

        self.camerapix   = camerapix
        self.obs_weights = obs_weights
        self.resMatrices = resMatrices
        self.idealJacs   = idealJacs
        self.J_S         = J_S
        
        super().__init__(J_S.dtype, (nBins, nFreeParms))
        

    # |v| = number of free parameters
    def _matvec(self, v):

        nBins = self.shape[0]
        
        w = np.empty(nBins, dtype=v.dtype)
        
        vFull = self.J_S.dot(v.ravel())
        
        for campix, iJac, resM in zip(self.camerapix, self.idealJacs, self.resMatrices):
            s = campix[0]
            e = campix[1]
            
            vJac = iJac._matvec(vFull)
            w[s:e] = resM.dot(vJac)
            
        return self.obs_weights * w

    # |v| = number of observable bins
    def _rmatvec(self, v):

        nFreeParms = self.shape[1]
        
        w = np.zeros(nFreeParms, dtype=v.dtype)
        v0 = v * self.obs_weights
        for campix, iJac, resM in zip(self.camerapix, self.idealJacs, self.resMatrices):
            s = campix[0]
            e = campix[1]

            vSub = v0[s:e]
            vRes = resM.T.dot(vSub)
            vJac = iJac._rmatvec(vRes)

            vFreeParms = self.J_S.T.dot(vJac)
            w += vFreeParms

        return w
