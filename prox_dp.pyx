
import logging
import time

import numpy as np
cimport numpy as np

from numpy import linalg

cimport cython
from cpython cimport bool

ctypedef np.float32_t dtype_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef sortOffDiagonalIndirect(np.ndarray[double, ndim=2] Xt):
    cdef np.ndarray[double, ndim=2] Xtmp = -np.abs(Xt)
    np.fill_diagonal(Xtmp, 1.0) #Ensures they are last
    return np.argsort(Xtmp)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef applyOrdering(np.ndarray[double, ndim=2] V, np.ndarray[Py_ssize_t, ndim=2] ordering):
    return V[np.arange(V.shape[1])[:,None], ordering] #row-wise


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.profile(False)
@cython.infer_types(False)
def proxSubDualDecomp(np.ndarray[double, ndim=2] Z, 
                      double beta, 
                      np.ndarray[double, ndim=1] adelta, 
                      np.ndarray[double, ndim=2] adeltaMat, 
                      props, 
                      np.ndarray[double, ndim=2] warmV=None):
    
    cdef int n = Z.shape[0]
    cdef double baseStepSize = props.get("proxDualDecompStepSizeBase", 0.9)
    cdef double stepMul = props.get("proxDualDecompStepSizeMul", 1.0)
    cdef int maxIters = props.get("proxDualDecompMaxIters", 800)
    cdef bool useGeometric = props.get("dualDecompUseGeometricStep", True)
    cdef double dualDecompEpsilon = props.get("dualDecompEpsilon", 1e-6)
    cdef bool useWarmStart = props.get("useWarmStart", False)
    cdef bool useAcceleration = props.get("accelerateProximal", False)
    cdef double momentum = props.get("momentum", 0.0)
    logger = logging.getLogger("search.l1.prox")
    
    cdef double stepSize = baseStepSize
    cdef double ub, sgn, dualGradNorm, tlast, t
    cdef int i, j, k, cRoot, newRoot, sortedPos, ub_pos
    cdef double timeTaken = 0
    cdef int groupMerges = 0

    cdef np.ndarray[double, ndim=2] weightedZ, weights, vDiff
    cdef np.ndarray[Py_ssize_t, ndim=2] sortedToOrigPos, origtoSortedPos

    cdef np.ndarray[double, ndim=2] X = np.empty((n,n))
    cdef np.ndarray[Py_ssize_t, ndim=1] parent = np.empty(n, dtype=int)
    cdef np.ndarray[double, ndim=1] ubound = np.empty(n)
    cdef np.ndarray[double, ndim=1] value = np.empty(n)
    cdef np.ndarray[Py_ssize_t, ndim=1] groupSize = np.empty(n, dtype=int)
        
    np.fill_diagonal(X, 0.0)

    # Dual variable for coupling constraint for each edge
    # V is anti-symmetric
    cdef np.ndarray[double, ndim=2] V

    cdef np.ndarray[double, ndim=2] XFin = np.empty((n,n))
    t = 1.0
    dualGradNorm = 0.0

    # X is not symmetric during the routine
    if warmV is not None and useWarmStart:
        V = warmV
    else:
        V = np.zeros((n,n))


    logger.debug("Finished initialization for prox")

    sTime = time.time()

    for iteration in range(maxIters):
        if useGeometric:
            stepSize *= stepMul
        else:
            stepSize = baseStepSize/(iteration + 1)
    
    
        # Sort order is w.r.t (Z-V)
        weightedZ = Z - V
        # 80% of the time of this method is spent in these (split evenly-ish)
        sortedToOrigPos = sortOffDiagonalIndirect(weightedZ)
        origtoSortedPos = np.argsort(sortedToOrigPos)
        weights = applyOrdering(adeltaMat, origtoSortedPos)
    
        # Process nodes one by one
        i = 0
        while i < n:
            ub = np.inf
            
            # Find optimal value for edge, within it's bounds
            sortedPos = 0
            while sortedPos < (n-1):
                j = sortedToOrigPos[i,sortedPos]
                
                if weightedZ[i,j] > 0:
                    sgn = 1
                else:
                    sgn = -1

                # Create singleton set
                parent[j] = j
                # negative value indicates opposite sign from weightedZ
                value[j] = sgn*weightedZ[i,j] - 2*beta*weights[i,j]
                ubound[j] = ub
                groupSize[j] = 1
                
                # Needs to be merged with group before it 
                if value[j] >= ubound[j]:
                    cRoot = j
                    
                    # Start repeated group merging procedure
                    while value[cRoot] >= ubound[cRoot]:
                        if cRoot != j:
                            groupMerges += 1
                        # Attach to new root
                        ub_pos = sortedToOrigPos[i,origtoSortedPos[i, cRoot]-1]
                        k = ub_pos
                        while parent[k] != k:
                            k = parent[k]
                        newRoot = k
                        parent[cRoot] = newRoot
                        
                        # Update group value and size attributes
                        value[newRoot] = (value[newRoot]*groupSize[newRoot] + 
                                          value[cRoot]*groupSize[cRoot])/(
                                         groupSize[newRoot] + groupSize[cRoot])
                        
                        groupSize[newRoot] += groupSize[cRoot]
                        
                        cRoot = newRoot
                
                        ub = value[newRoot]
                else:
                    ub = value[j]
                  
                sortedPos += 1 
                
            # Update each X[i,j] to its root's value, but with it's own sign  
            j = n-1
            while j >= 0:
                if i != j:
                    k = j
                    while parent[k] != k:
                        k = parent[k]
                    X[i,j] = value[k]
                    if X[i,j] < 0:
                        X[i,j] = 0.0
                    if weightedZ[i,j] < 0:
                        X[i,j] = -X[i,j] # Set to correct sign
                j -= 1            
            i += 1
                        
        vDiff = X - X.T
        
        V += stepSize*vDiff
        dualGradNorm = linalg.norm(vDiff)
        
        if iteration % 10 == 0:
            logger.debug("Iteration %d, dual gnorm: %1.8f, %1.2e, eps: %1.1e, t=%2.3f",
                        iteration, dualGradNorm, dualGradNorm, dualDecompEpsilon, t)
            groupMerges = 0
                    
        if dualGradNorm < dualDecompEpsilon:
            logger.debug("Terminating dual decomposition due to convergence at iter %d", iteration)
            break
    
    eTime = time.time()
    timeTaken = eTime-sTime
    
    logger.info("Dual Decomp finished, eps=%1.2e, iter %d, dual gnorm: %2.5f, iter-t: %1.2e", 
                dualDecompEpsilon, iteration, dualGradNorm, timeTaken/(iteration+1))
    
    XFin = 0.5*(X + X.T)
    
    # Ensure zeros are, ah, zero
    for i in range(n):
        for j in range(n):
            if abs(X[i,j]) < 1e-10 or abs(X[j,i]) < 1e-10:
                XFin[i,j] = 0.0
        XFin[i,i] = Z[i,i]
    
    if n <= 10:
        logger.info("Z:\n%s", Z)
        logger.info("weightedZ:\n%s", weightedZ)
        logger.info("X:\n%s", XFin)
        logger.info("V:\n%s", V)
        logger.info("Vdiff:\n%s", vDiff)
    return XFin

