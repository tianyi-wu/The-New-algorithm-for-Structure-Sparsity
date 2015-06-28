
import logging
import scipy
from prox_dp import *
from numpy import *
import numpy.random
import scipy.linalg
from scipy.linalg import fblas as FB

def shrink(A, rho, regDiagonal):
    shrunk = sign(A)*maximum(abs(A) - rho, 0.0)
    if not regDiagonal:
        fill_diagonal(shrunk, diagonal(A))
    return shrunk
    

## Find solution using ADMM method
# For L1 methods, A is a weight matrix or scalar for the prior
# For degree prior, A is a 1d array of the degree weights
def admmDP(C, a=1.0, props={}, warmX=None, warmU=None):
    n = C.shape[0]
    logger = logging.getLogger("search.admm.dp")
    beta = props.get('priorWeight', 0.01)
    maxIter = props.get("maxParamaterOptIters", 1000)
    epsilon = props.get("epsilon", 1e-6)
    returnIntermediate = props.get("returnIntermediate", False)
    regDiagonal = props.get("regDiagonal", False)
    degreePriorADMM = props.get("degreePrior", True)
    adaptiveMu = props.get("adaptiveMu", True)
    props['dualDecompEpsilon'] = props.get('dualDecompEpsilon', 1e-10)
    sTime = time.time()

    logger.info("Starting ADMM learning. n=%d beta=%1.4f", n, beta)

    mu = props.get("mu", 0.1) #10.0
    muChange = props.get("muChange", 0.1)
    
    if props.get("normalizeForADMM", True):
        logger.info("-- NORMALIZING C --")
        covNormalizer = sqrt(diagonal(C))
        C = C / outer(covNormalizer, covNormalizer)
    else:
        logger.info("NOT NORMALIZING C")
    
    # Rescale to make beta's range a better fit
    maxOffDiag = numpy.max(numpy.abs(tril(C, -1)))
    C = array(C / maxOffDiag)
    
    if warmX is not None:
        X = warmX
    else:
        X = eye(n)
        
    if warmU is not None:
        U = warmU
    else:
        U = eye(n)
    
    
    Z = copy(X)
    
    ll = inf
    Xs = []
    gs = []
    ds = []

    if degreePriorADMM:
        adelta = -a[:n] + a[1:]
        adeltaMat = outer(ones(n), adelta)
        beta = beta / 2

        logger.error("adelta: %s", adelta[:6])

    warmV = zeros((n,n))

    if a is None:
        a = 1.0

    for i in range(maxIter):
            
        #####################################################################
        ##### Eigenvalue update to X
        logger.debug("Performing eigenvalue decomposition")
        
        for retry in range(6):
            try:
                A = mu*(Z - U) - C
                (lamb, Q) = linalg.eigh(A)
                logger.debug("Decomposition finished")
                break
            except numpy.linalg.linalg.LinAlgError as err:
                # If A is not in the PSD cone, we reduce the step size mu
                logger.error("Failed eigendecomposition with mu=%2.2e", mu)
                mu *= 0.5
                U /= 0.5
                logger.error("Retry %d, halving mu to: %2.5f", retry, mu)
        
        newEigs = (lamb + sqrt(lamb*lamb + 4*mu)) / (2*mu)
        X = FB.dgemm(alpha=1.0, a=(Q*newEigs), b=Q, trans_b=True)
        
        #### Soft thresholding update Z
        logger.debug("Starting Proximal step")
        Zpreshrink = X + U
        Zlast = copy(Z)
        if degreePriorADMM:
            Z = proxSubDualDecomp(Zpreshrink, beta/mu, adelta, adeltaMat, props, warmV)
        else:
            Z = shrink(Zpreshrink, beta*a/mu, regDiagonal)
        
        if props.get("nonpositivePrecision", False):
            Z = Z * (Z < 0) + diag(diag(Z))
        
        ### Update U ( U is the sum of residuals so far )
        logger.debug("Updating U")
        U += X - Z
        #####################################################################
        
        dualResidual = linalg.norm(Z - Zlast)
        residual = linalg.norm(X-Z)
           
        if adaptiveMu:
            # if the two residuals differ my more than this factor, adjust mu (p20)
            differenceMargin = 10
            if residual > dualResidual*differenceMargin:
                mu *= 1.0 + muChange
                U /= 1.0 + muChange
                logger.debug("*** Increasing mu to %2.6f", mu)
            elif dualResidual > residual*differenceMargin:
                mu *= 1.0 - muChange
                U /= 1.0 - muChange
                logger.debug("*** Decreasing mu to %2.6f", mu)
        
        # Ensure that the dual decomp procedure is run with enough accuracy
        ddeps = props['dualDecompEpsilon']
        margin = 50.0
        if residual < margin*ddeps or dualResidual < margin*ddeps:
            props['dualDecompEpsilon'] = min(residual, dualResidual)/margin
            
        if returnIntermediate:
            ds.append(dualResidual)
            gs.append(residual)
        
        if residual < epsilon and dualResidual < epsilon:
            logger.info("Converged to %2.3e in %i iters", residual, i+1)
            break
        
        edges = (count_nonzero(Z) - n) / 2
        logger.info("Iter %d, res: %2.2e, dual res: %2.2e, mu=%1.1e, %d edges free", 
                    i+1, residual, dualResidual, mu, edges)
    
    eTime = time.time()
    timeTaken = eTime-sTime
    logger.info("Time taken(s): %5.7f", timeTaken)
    
    if residual > epsilon or dualResidual > epsilon:
        logger.error("NONCONVERGENCE!!, res: %2.2e, dres: %2.2e, iters: %d", 
                    residual, dualResidual, i)
    
    edges = (count_nonzero(Z) - n) / 2
    logger.info("regDiagonal: %s, beta: %2.4f", regDiagonal, beta)
    logger.info("Edges %d out of %d   | eps=%1.1e", edges, (n*n - n)/2, epsilon)
    logger.info("Final residual=%2.2e, dual res=%2.2e", residual, dualResidual)
            
    return {'X': Z, 'U': U, 'obj': ll, 'iteration': i+1, 'Xs': Xs, 'gs': gs, 'ds': ds,
            'timeTaken': timeTaken, 'edges': edges, 'Zpreshrink': Zpreshrink, 'bm': beta/mu}
