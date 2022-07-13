import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
import jax.random as random
import jax
import scipy.optimize as opt

K = 100
h = 2/(K-1)
x = -1 + np.arange(K)*h

def func(y):
    # going to need to compute 2 integrals
    yEqn = np.empty(K)

    # for all equations y_k
    for k in range(K):
    
        # calculate green's function
        G1 = (x[k]-1)*(x[0:k+1]+1)/2
        G2 = (x[k]+1)*(x[k:K]-1)/2
        #G = np.empty(K+1)
        #G[0:k+1] = np.multiply(x[k]-1, x-1)/2
        #G[k+1:K+1] = np.multiply(x[k]+1, x-1)/2
        
        # calculate the other part of the integrand
        yPart = y + np.power(y, 2)
        
        # calculate the entire integrand
        f1 = np.multiply(G1, yPart[0:k+1])
        f2 = np.multiply(G2, yPart[k:K])
        
        # compute both integrals using trapezoid rule
        integral1 = np.multiply(h/2, np.sum(2*f1[1:-1]) + f1[0] + f1[len(f1)-1])
        integral2 = np.multiply(h/2, np.sum(2*f2[1:-1]) + f2[0] + f2[len(f2)-1])
        
        yEqn[k] = integral1 + integral2
        #yEqn[k] = (np.power(x[k],2)-1)/2 - integral1 - integral2

    yEqn = (np.power(x,2) - 1)/2 - yEqn
    return yEqn
'''
def jac(y):
    j = np.diag(y*2*h*h + h*h - 2)
    j += np.diag(np.ones(K-1), k=1)
    j += np.diag(np.ones(K-1), k=-1)

    print('condition number: ', np.linalg.cond(j))
    return j
'''

y0 = np.full(K,-1)
#y0 = (x+1)*(x-1)
infodict = opt.fsolve(func, y0, full_output=True)
#infodict = opt.fsolve(func, y0, fprime=jac, full_output=True)
y = infodict[0]

#plt.plot(np.arange(len(y)), y)
plt.plot(np.arange(len(y)), y)
plt.show()



