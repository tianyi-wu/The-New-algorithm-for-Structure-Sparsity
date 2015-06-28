
import scipy
from numpy import *
import numpy.linalg
    
def logDegreeWeights(n, offset, linearSlope):
    inds = array(range(n)) + offset
    lTerm = array(range(n))*linearSlope
    return log(inds) + lTerm
    
def rootDegreeWeights(n, power, linearSlope):
    inds = array(range(n)) + 1
    lTerm = array(range(n))*linearSlope
    return pow(inds, 1.0/power) - 1 + lTerm
    
def linearDegreeWeights(n):
    return array(range(n))

def logNomarlWeights(n, mu,sigma,offset, linearSlope):
    inds = array(range(n)) + offset
    lTerm = array(range(n))*linearSlope
    return log(inds) + (((log(inds)-mu)/sigma)**2)/2 + lTerm