import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
import jax.random as random
import jax
import scipy.optimize as opt

K = 20
h = 2/(K-1)
x = -1 + np.arange(K)*h

def func(y):
    # going to need to compute 2 integrals
    yEqn = np.empty(K)

    # for all equations y_k
    for k in range(K):
        integral1 = computeIntegral1(x,y,k,K)
        integral2 = computeIntegral2(x,y,k,K)
        
        yEqn[k] = integral1 + integral2

    yEqn = (np.multiply(x,x) - 1)/2 - yEqn

    return yEqn

def computeIntegral1(x, y, k, K):
        xbound = k+1
        G = (x[k]-1)*(x[0:xbound]-1)/2
        yPart = y[0:xbound] + np.power(y[0:xbound], 2)
        f = np.multiply(G, yPart)
        
        if (k > 0):
            i = np.arange(1,xbound)
            deltaX = x[i] - x[i-1]
            deltaF = f[i] + f[i-1]

            return np.sum(np.multiply(deltaX, deltaF)/2)
        else:
            return 0
 
def computeIntegral2(x, y, k, K):
        xbound = k+1
        G = (x[k]+1)*(x[xbound:K]-1)/2
        yPart = y[xbound:K] + np.power(y[xbound:K], 2)
        f = np.multiply(G, yPart)
        #integral2 = (h/2)*(f2[0] + 2*np.sum(f2[1:K-2]) + f2[K-1]) 
        
        i = np.arange(xbound,K)
        deltaX = x[i] - x[i-1]

        i = np.arange(K-xbound)
        deltaF = f[i] + f[i-1]

        return np.sum(np.multiply(deltaX, deltaF)/2)
 

y0 = np.full(shape=K, fill_value=0)
y = opt.fsolve(func, y0, full_output=False)

plt.plot(np.arange(len(y)), y)
plt.show()



