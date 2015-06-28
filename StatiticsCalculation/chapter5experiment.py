
import logging
import logging.config
import time
import scipy
from admm_dp import *
from degree_weights import *
from edge_correctness import *
from numpy import *

import pylab as plt


# Make matrices readable:
set_printoptions(precision=3, suppress=True, linewidth=100)

#### Setup logging ####
logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
logger = logging.getLogger("opt")

#mats = load('testMatrix1.npz')
#mats = load('BAmodel100.npz')
mats = load('DegreeSequence100.npz')

cov = mats['cov']
#cov = mats['BCov']

#A = mats['B']
A = mats['A'] # True edge structure indicator matrix

n = cov.shape[0]


#alog = logNomarlWeights(n=n+1, mu=1.3047, sigma=4, offset=2, linearSlope=2.1)
#aroot = rootDegreeWeights(n=n+1, power=2.0, linearSlope=2.1)
#alinear = linearDegreeWeights(n=n+1)

#a = [alog,aroot,alinear]
#F = np.zeros([3,3])

props = {
     'priorWeight': 0.20 ,#0.8, # Change to get desired number of edges
     'maxParamaterOptIters': 800,
     
     # If the method doesn't converge, mu may need to be changed
     # to a different value. It attempts to tune it automatically,
     # but it doesn't always work.
     'adaptiveMu': True,
     'mu': 0.5,
}

#'priorWeight': 0.20


#PriorWeight = np.array(range(1,31))/40.0
#PriorWeight = np.array(range(15,23))/80.0
PriorWeight = np.array(range(45,69))/240.0
l = PriorWeight.size

alog = logNomarlWeights(n=n+1, mu=1.3047, sigma=4, offset=2, linearSlope=0.8)
logResult = np.zeros([2,l])

for p in range(l):
    props['priorWeight'] = PriorWeight[p]
    props["degreePrior"]=True

    results = admmDP(cov, alog, props)
    X = results['X']
    (edges, truePositives, falsePositives, falseNegatives, trueNegatives) = edgeCorrectness(X,A, logger)
    logResult[0,p] = falsePositives*1.0/ (trueNegatives+falsePositives)
    logResult[1,p] = truePositives*1.0/(truePositives+falseNegatives)



#PriorWeight = np.array(range(7,12))/80.0

PriorWeight = np.array(range(21,36))/240.
l = PriorWeight.size

aroot = rootDegreeWeights(n=n+1, power=2.0, linearSlope=2.0)
rootResult = np.zeros([2,l])



for p in range(l):
    props['priorWeight'] = PriorWeight[p]
    props["degreePrior"]=True
    results = admmDP(cov, aroot, props)
    X = results['X']
    (edges, truePositives, falsePositives, falseNegatives, trueNegatives) = edgeCorrectness(X,A, logger)
    rootResult[0,p] = falsePositives*1.0/ (trueNegatives+falsePositives)
    rootResult[1,p] = truePositives*1.0/(truePositives+falseNegatives)




#PriorWeight = np.array(range(11,18))/60.0
PriorWeight = np.array(range(33,56))/180.0

l = PriorWeight.size
linearResult = np.zeros([2,l])

for p in range(l):
    props['priorWeight'] = PriorWeight[p]
    props["degreePrior"]=False
    results = admmDP(cov, props=props)
    X = results['X']
    (edges, truePositives, falsePositives, falseNegatives, trueNegatives) = edgeCorrectness(X,A, logger)
    linearResult[0,p] = falsePositives*1.0/ (trueNegatives+falsePositives)
    linearResult[1,p] = truePositives*1.0/(truePositives+falseNegatives)

#linestyle = '--'
plt.plot(logResult[0,:],logResult[1,:], color="blue", label="lognormal")
plt.plot(rootResult[0,:],rootResult[1,:], color="green", label="root")
plt.plot(linearResult[0,:],linearResult[1,:], color="red",label="L1")

plt.legend(loc='lower right')

#plt.title(' ROC curve in BA Model')
plt.title(' ROC curve in PD Model')

plt.xlabel('False Positives in all negatives')
plt.ylabel('True Positives in all positives')





# for i in range(3):
#     if i == 2:
#         props["degreePrior"]=False
#         results = admmDP(cov, props=props)
#     else:
#         results = admmDP(cov, a[i], props)
    
#     X = results['X']
#     (edges, truePositives, falsePositives, falseNegatives) = edgeCorrectness(X,A, logger)
#     P = truePositives*1.0/(truePositives+falsePositives)
#     F[i,0] = P
#     R = truePositives*1.0/ (truePositives+falseNegatives)
#     F[i,1] = R
#     F[i,2] = 2.0*P*R/(P+R)

