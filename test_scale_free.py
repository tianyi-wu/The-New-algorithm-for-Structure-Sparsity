
import logging
import logging.config
import time
import scipy
from admm_dp import *
from degree_weights import *
from edge_correctness import *
from numpy import *

import networkx as nx

props = {
     'priorWeight': 0.19 ,#0.8, # Change to get desired number of edges
     'maxParamaterOptIters': 800,
     
     # If the method doesn't converge, mu may need to be changed
     # to a different value. It attempts to tune it automatically,
     # but it doesn't always work.
     'adaptiveMu': True,
     'mu': 0.5,
}

# Make matrices readable:
set_printoptions(precision=3, suppress=True, linewidth=100)

#### Setup logging ####
logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
logger = logging.getLogger("opt")

mats = load('BAmodel100.npz')
#mats = load('scale-free-20.npz')
#mats = load('DegreeSequence20.npz')
#mats = load('DegreeSequence100.npz')


cov = mats['cov']
#cov = mats['BCov']

#A = mats['B']
A = mats['A'] # True edge structure indicator matrix

n = cov.shape[0]

aroot = rootDegreeWeights(n=n+1, power=2.0, linearSlope=1.85)


#a = logDegreeWeights(n=n+1, offset=2, linearSlope=2.1)
#a = linearDegreeWeights(n=n+1)

alog = logNomarlWeights(n=n+1, mu=1.3047, sigma=4, offset=2, linearSlope=1.8)


#logger.info("a: %s", a[:10])

# ### Run ####
# props["degreePrior"]=False
# props['priorWeight']= 0.50 
# results = admmDP(cov, a=a,props=props)

# X = results['X']
# (edges, truePositives, falsePositives, falseNegatives) = edgeCorrectness(X,A, logger)
# P = truePositives*1.0/(truePositives+falsePositives)
# R = truePositives*1.0/ (truePositives+falseNegatives)
# f = 2.0*P*R/(P+R)

#G=nx.from_numpy_matrix(X)
#pos=nx.spring_layout(G)
#nx.draw(G, pos =pos, with_labels=True, node_color='white')

a = [alog,aroot]
F = np.zeros([3,3])
props["degreePrior"]=True
for i in range(3):
    if i == 2:
        props["degreePrior"]=False
        results = admmDP(cov, props=props)
    else:
        results = admmDP(cov, a[i], props)
    
    X = results['X']
    (edges, truePositives, falsePositives, falseNegatives, trueNegatives)= edgeCorrectness(X,A, logger)
    P = truePositives*1.0/(truePositives+falsePositives)
    F[i,0] = P
    R = truePositives*1.0/ (truePositives+falseNegatives)
    F[i,1] = R
    F[i,2] = 2.0*P*R/(P+R)
