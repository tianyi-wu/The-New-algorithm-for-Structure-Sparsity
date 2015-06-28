import logging
import logging.config
import time
import scipy
from admm_dp import *
from degree_weights import *
from edge_correctness import *
from numpy import *

# Make matrices readable:
set_printoptions(precision=3, suppress=True, linewidth=100)

#### Setup logging ####
logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
logger = logging.getLogger("opt")
logger.setLevel(logging.DEBUG)

mats = load('testMatrix1.npz')
#mats = load('scale-free-20.npz')
#mats = load('DegreeSequence20.npz')


cov = mats['cov']
#cov = mats['BCov']

A = mats['A'] # True edge structure indicator matrix
#A = mats['B']

n = cov.shape[0]

#a = rootDegreeWeights(n=n+1, power=2.0, linearSlope=0.1)
#a = logDegreeWeights(n=n+1, offset=0.5, linearSlope=0.5)
#a = logDegreeWeights(n=n+1, offset=10, linearSlope=0.5)
#a = linearDegreeWeights(n=n+1)
#
# 0.32 ,  2.   ,  1.
props = {
     'priorWeight': 0.17, #3, # Change to get desired number of edges
     'maxParamaterOptIters': 800,
     #'epsilon' : 1e-4,
     #'dualDecompEpsilon' : 1e-8,
     #"muChange" : 0.5,
     #'accelerateProximal': True,
     #"degreePrior": False,
     #'normalizeForADMM' : False,
     # to a different value. It attempts to tune it automatically,
     # but it doesn't always work.
     'adaptiveMu': True,
     'mu': 0.5
     #'mu': 0.1
} 



#FValue = np.array(rootDegreeWeights(n+1, 3.0, 0.5)) -rootDegreeWeights(n+1, 3.0, 0.5)
#PriorWeight = array([0.03,0.06,0.12,0.24,0.48,0.96,1.92,3.84,7])
PriorWeight = array([0.08,0.11,0.14,0.17,0.20,0.24])

#Epsilon = array([2,2.5,3,3.5,4])
Beta = array([1.5,1.8,2.1,2.4])
Sigma = array([3.2,3.5,4.0,4.5,5.0])


Result = zeros([PriorWeight.size * Sigma.size * Beta.size,6])
i=0
for alpha in PriorWeight:
     for b in Beta:
          for s in Sigma:
               props['priorWeight'] = alpha
               a = logNomarlWeights(n=n+1, mu=1.3047, sigma=4, offset=2, linearSlope=b)

               results = admmDP(cov, a, props)
               X = results['X']
               (edges, truePositives, falsePositives, falseNegatives) = edgeCorrectness(X,A, logger)
               #Result[i,5] = truePositives
               #Result[i,6] = falsePositives
               #Result[i,7] = falseNegatives

               P = truePositives*1.0/(truePositives+falsePositives)
               R = truePositives*1.0/ (truePositives+falseNegatives)
               f = 2.0*P*R/(P+R)
               Result[i,0] = P
               Result[i,1] = R
               Result[i,2] = f
               Result[i,3] = alpha
               Result[i,4] = b
               Result[i,5] = s
               i+=1






