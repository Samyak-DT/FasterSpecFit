#
# Compute a mapping from the free parameters of a spectrum fitting problem
# to the full set of parameters for the EMLine model, as well the Jacobian
# of this mapping.  We capture most of the complexity of the mapping once and
# do the minimum necessary updating to it for each new set of freeParameters.
#

import numpy as np

from numba import jit

class ParamsMapping(object):

    def __init__(self, fixedParameters, freeParms,
                 tiedParms, tiedSources, tiedFactors,
                 doubletRatios, doubletSources):
        
        self.nParms     = len(fixedParameters)
        self.nFreeParms = len(freeParms)
        
        # permutation mapping each free parameter in full list
        # to its location in free list
        p = np.empty(self.nParms, dtype=np.int32)
        p[freeParms] = np.arange(self.nFreeParms, dtype=np.int32)

        self._precomputeMapping(fixedParameters, freeParms,
                                tiedParms, tiedSources, tiedFactors,
                                doubletRatios, doubletSources,
                                p)
        
        self._precomputeJacobian(freeParms,
                                 tiedParms, tiedSources, tiedFactors,
                                 doubletRatios, doubletSources,
                                 p)
        

    #
    # mapFreeToFull()
    # Given a vector of free parameters, return the corresponding
    # list of full parameters, accounting for fixed, tied, and
    # doublet features.
    #
    def mapFreeToFull(self, freeParms):

        return self._mapFreeToFull(freeParms,
                                   self.nParms,
                                   self.sources,
                                   self.factors,
                                   self.doubletPatches)

    #
    # _mapFreeToFull()
    # Given a vector of free parameters, return the corresponding
    # list of full parameters, accounting for fixed, tied, and
    # doublet features.
    #
    @staticmethod
    @jit(nopython=True, fastmath=True, nogil=True)
    def _mapFreeToFull(freeParms, nParms, sources, factors, doubletPatches):
        
        for i, src_j_free in doubletPatches:
            factors[i] = freeParms[src_j_free]

        fullParms = np.empty(nParms, dtype=freeParms.dtype)

        for i, src in enumerate(sources):
            fullParms[i] = factors[i] # copy fixed value
            if src != -1:
                fullParms[i] *= freeParms[src]
        
        return fullParms
        
    
    
    #
    # getJacobian()
    # Given a vector v of free parameters, return the Jacobian
    # of the transformation from free to full at v.  The Jacobian
    # is a sparse matrix represented as an array of nonzero entries
    # (jacElts) and their values (jacFactors)
    #
    def getJacobian(self, freeParms):
        for i, j_free, src_j_free in self.jacDoubletPatches:
            self.jacFactors[i]   = freeParms[src_j_free]
            self.jacFactors[i+1] = freeParms[j_free]
        
        return ((self.nParms, self.nFreeParms), self.jacElts, self.jacFactors)
    

    #
    # Multiply parameter Jacobian J_S * v, writing result to w.
    #
    @staticmethod
    @jit(nopython=True, fastmath=True, nogil=True)
    def _matvec(J_S, v, w):
    
        shape, elts, factors = J_S

        for j in range(shape[0]): # total params
            w[j] = 0.
        
        for i, (dst, src) in enumerate(elts):
            w[dst] += factors[i] * v[src]
            


    #
    # Multiply parameter Jacobian v * J_S^T, *adding* result to w.
    #
    @staticmethod
    @jit(nopython=True, fastmath=True, nogil=True)
    def _rmatvec(J_S, v, w):

        _, elts, factors = J_S
            
        for i, (dst, src) in enumerate(elts):
            w[src] += factors[i] * v[dst]

    
    ###########################################################
    
    #
    # Precompute all the transformations from free parameters
    # to full parameters that do not require knowledge of the
    # free parameter values.
    #
    def _precomputeMapping(self, fixedParameters, freeParms,
                           tiedParms, tiedSources, tiedFactors,
                           doubletRatios, doubletSources,
                           p):
        
        # by default, assume parameters are fixed and that
        # they take on the values in fixedParameters
        sources = np.full(self.nParms, -1, dtype=np.int32)
        factors = fixedParameters.copy()

        for j in freeParms:
            sources[j] = p[j]
            factors[j] = 1.

        for j, src_j, factor in zip(tiedParms, tiedSources, tiedFactors):
            if src_j not in freeParms:
                #print(f"SOURCE {src_j} tied to {j} is not free!")
                # if source is fixed, so is target, and it's in fixedParameters
                pass
            else:
                sources[j] = p[src_j]
                factors[j] = factor

        doubletPatches = []
        for (j, src_j) in zip(doubletRatios, doubletSources):
            #if j not in freeParms:
            #    print(f"ratio {j} in doublet with {src_j} is not free!")
            #if src_j not in freeParms:
            #    print(f"amplitude {src_j} in doublet with {j} is not free!")

            if j not in freeParms or src_j not in freeParms:
                continue

            sources[j] = p[src_j]  # factor will be patched dynamically
            doubletPatches.append((j, p[j]))

        self.sources = sources
        self.factors = factors
        self.doubletPatches = np.array(doubletPatches)
        
        
    #
    # Precompute as much of the Jacobian of the transformation
    # from free parameters to full parameters as does not require
    # knowledge of the free parameter values.
    #
    def _precomputeJacobian(self, freeParms,
                            tiedParms, tiedSources, tiedFactors,
                            doubletRatios, doubletSources,
                            p):
        jacElts = []
        
        # free params present unchanged in full list
        for j in freeParms:
            jacElts.append((j, p[j], 1.))
            
        # tied params
        for j, src_j, factor in zip(tiedParms, tiedSources, tiedFactors):
            if src_j not in freeParms:
                #print(f"SOURCE {src_j} tied to {j} is not free!")
                # if source is fixed, so is target, so Jacobian contrib is 0
                pass
            else:
                jacElts.append((j, p[src_j], factor))
                
        # create placeholders for doublets in sparse structure
        # and record ops needed to compute Jacobian for given free params
        jacDoubletPatches = []
        for j, src_j in zip(doubletRatios, doubletSources):
            #if j not in freeParms:
            #    print(f"ratio {j} in doublet with {src_j} is not free!")
            #if src_j not in freeParms:
            #    print(f"amplitude {src_j} in doublet with {j} is not free!")

            if j not in freeParms or src_j not in freeParms:
                continue
            
            jacElts.append((j, p[src_j], 0.)) # will set factor to v[j]
            jacElts.append((j, p[j],     0.)) # will set fctor to v[src_j]

            # record enough info to update values of above two elts given
            # free paramter vector
            jacDoubletPatches.append((len(jacElts) - 2, p[src_j], p[j]))

        self.jacElts   = np.empty((len(jacElts), 2), dtype=np.int32)
        self.jacFactors = np.empty(len(jacElts))
        
        for i, tup in enumerate(jacElts):
            dst, src, factor = tup
            self.jacElts[i] = (dst, src)
            self.jacFactors[i] = factor
        
        self.jacDoubletPatches = np.array(jacDoubletPatches)



