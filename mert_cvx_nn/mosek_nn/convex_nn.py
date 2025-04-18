import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import scipy.io as sio

np.random.seed(0)
def relu(x):
    return np.maximum(0,x)

def drelu(x):
    return x>=0
path='data/Boston/' #'ConvexNN/mosek_nn/data/Boston'
X=np.loadtxt(path+'Xtrain.txt')
Xtest=np.loadtxt(path+'Xtest.txt')
y=np.loadtxt(path+'Ytrain.txt')
ytest=np.loadtxt(path+'Ytest.txt')
X=np.append(X,np.ones((X.shape[0],1)),axis=1)
Xtest=np.append(Xtest,np.ones((Xtest.shape[0],1)),axis=1)
n,d=X.shape
ntest,d=Xtest.shape

def solve_problem():
    betavec=np.array([0, 1, 5e-1, 1e-1, 1e-2, 1e-3, 1e-4])
    err_test=[]
    ## Approximation of all possible sign patterns
    mh=2000
    U1=np.random.randn(d,mh)
    dmat=drelu(X@U1)
    dmat, ind=(np.unique(dmat,axis=1, return_index=True))
    U=U1[:,ind]

    err_test=[]
    # CVXPY variables
    m1=dmat.shape[1]
    W1=cp.Variable((d,m1))
    W2=cp.Variable((d,m1))
    ## parameters
    yopt1=cp.Parameter((n,1))
    yopt2=cp.Parameter((n,1))    
    regw=cp.Parameter((1))
    # output componenp.nts
    yopt1=cp.sum(cp.multiply(dmat,(X@W1)),axis=1)
    yopt2=cp.sum(cp.multiply(dmat,(X@W2)),axis=1)
    # regularizations
    regw=cp.mixed_norm(W1.T,2,1)+cp.mixed_norm(W2.T,2,1)
    # constructs the optimization problem
    betaval = cp.Parameter(nonneg=True)
    cost=cp.sum(cp.sum_squares(y-(yopt1-yopt2)))/(n)+betaval*regw
    constraints=[]
    constraints+=[cp.multiply((2*dmat-np.ones((n,m1))),(X@W1))>=0]
    constraints+=[cp.multiply((2*dmat-np.ones((n,m1))),(X@W2))>=0]
    # Mosek parameters dictionary
    params = {
      "MSK_IPAR_NUM_THREADS": 8,
      #"MSK_IPAR_INTPNT_MAX_ITERATIONS": 10,
      #"MSK_IPAR_OPTIMIZER": 0 # auto 0, interior point 1, conic 2
      #"MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-2
      #"MSK_DPAR_INTPNT_TOL_PSAFE": 0.01
      #"MSK_IPAR_OPTIMIZER": "free"
      #"MSK_IPAR_INTPNT_SOLVE_FORM": 1
      }
    # solve the problem
    prob=cp.Problem(cp.Minimize(cost),constraints)
    for i in range(betavec.shape[0]):
        betaval.value = betavec[i]
        prob.solve(solver=cp.MOSEK,warm_start=True,verbose=True,mosek_params=params)
        cvx_opt=prob.value
        print("Status: ",prob.status)
        W1v=W1.value
        W2v=W2.value
        ytest_est=np.sum(drelu(Xtest@U)*(Xtest@W1v)-drelu(Xtest@U)*(Xtest@W2v),axis=1)
        err=np.linalg.norm(ytest_est-ytest)**2/ntest
        err_test.append(err)
        print("Test error: ", err_test)
    print("Minimum test error: ", np.min(err_test))
    return np.min(err_test)
    np.savetxt('W1.txt', W1v, delimiter=',') 
    np.savetxt('testerr.txt',err_test,delimiter=',')
solve_problem() # train
