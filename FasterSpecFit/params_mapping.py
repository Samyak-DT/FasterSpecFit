import numpy as np

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
        self.doubletPatches = doubletPatches
        
            
    def _precomputeJacobian(self, freeParms,
                            tiedParms, tiedSources, tiedFactors,
                            doubletRatios, doubletSources,
                            p):
        jacOps = []
        
        # free params present unchanged in full list
        for j in freeParms:
            jacOps.append((j, p[j], 1.))
            
        # tied params
        for j, src_j, factor in zip(tiedParms, tiedSources, tiedFactors):
            if src_j not in freeParms:
                #print(f"SOURCE {src_j} tied to {j} is not free!")
                # if source is fixed, so is target, so Jacobian contrib is 0
                pass
            else:
                jacOps.append((j, p[src_j], factor))
                
        # create placeholders for doublets in sparse structure
        # and record ops needed to compute Jacobian for given free params
        jacDoubletPatches = []
        for (j, src_j) in zip(doubletRatios, doubletSources):
            #if j not in freeParms:
            #    print(f"ratio {j} in doublet with {src_j} is not free!")
            #if src_j not in freeParms:
            #    print(f"amplitude {src_j} in doublet with {j} is not free!")

            if j not in freeParms or src_j not in freeParms:
                continue
            
            jacOps.append((j, p[src_j], 0.)) # will set factor to v[j]
            jacOps.append((j, p[j],     0.)) # will set fctor to v[src_j]

            jacDoubletPatches.append((len(jacOps) - 2, p[src_j], p[j]))

        self.jacOps     = np.empty((len(jacOps), 2), dtype=np.int32)
        self.jacFactors = np.empty(len(jacOps))
        
        for i, tup in enumerate(jacOps):
            dst, src, factor = tup
            self.jacOps[i] = (dst, src)
            self.jacFactors[i] = factor

        self.jacDoubletPatches = jacDoubletPatches


    def mapFreeToFull(self, freeParms):
        for i, src_j_free in self.doubletPatches:
            self.factors[i] = freeParms[src_j_free]

        # FIXME: this is slow
        fullParms = np.empty(self.nParms, dtype=freeParms.dtype)
        for i in range(self.nParms):
            if self.sources[i] == -1:
                fullParms[i] = self.factors[i] # copy fixed value
            else:
                fullParms[i] = freeParms[self.sources[i]] * self.factors[i]
        return fullParms
        
    # evaluate Jacobian at v = freeParms
    def getJacobian(self, freeParms):
        for i, j_free, src_j_free in self.jacDoubletPatches:
            self.jacFactors[i]   = freeParms[src_j_free]
            self.jacFactors[i+1] = freeParms[j_free]
                    
        return ((self.nParms, self.nFreeParms), self.jacOps, self.jacFactors)

