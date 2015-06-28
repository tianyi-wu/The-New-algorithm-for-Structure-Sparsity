

import logging
import time
import numpy
from numpy import *
import numpy.linalg

def edgeCorrectness(X, A, logger):
    threshold = 1e-10
    n = A.shape[0]
    Xind = abs(X) > threshold
    fill_diagonal(Xind, True)
    Aind = abs(A) > threshold
    fill_diagonal(Aind, True)
    
    # Check if diagonal contains zeros:
    nz = count_nonzero(diagonal(X))
    if nz != n:
        raise Exception("Zero on diagonal: %d" % (n-nz))
    
    # check symmetry
    if any(Xind != Xind.T):
        logger.error("X NOT SYMMETRIC!!!!!!!!!!!!!!!!!!!!!")
        raise Exception("Bad X: %1.1e" % linalg.norm(X - X.T))
    if any(Aind != Aind.T):
        logger.error("A NOT SYMMETRIC!!!!!!!!!!!!!!!!!!!!!")
        raise Exception("Bad A") 
        
    edges = (count_nonzero(Xind) - n) / 2
    
    truePositives = (sum(logical_and(Xind, Aind)) - n) / 2
    falsePositives = sum(logical_and(Xind, logical_not(Aind))) / 2
    falseNegatives = sum(logical_and(logical_not(Xind), Aind)) / 2
    trueNegatives = sum(logical_and(logical_not(Xind), logical_not(Aind))) / 2

    logger.info("TP: %d, FP: %d, FN: %d, edges %d", truePositives, 
                falsePositives, falseNegatives, edges)
                
    return (edges, truePositives, falsePositives, falseNegatives, trueNegatives)
