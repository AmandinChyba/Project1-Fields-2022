import jax.numpy as np
import jax.random as random
import scipy.optimize as opt
import matplotlib.pyplot as plt
import numpy

K = 40
h = 2/(K+1)

def func(y):
    k = numpy.arange(1, K-1)
    eqns = numpy.empty(K)
    
    eqns[0] = y[1] - 2*y[0] + y[0]*h*h + y[0]*y[0]*h*h - h*h
    eqns[k] = y[k+1] - 2*y[k] + y[k-1] + y[k]*h*h + y[k]*y[k]*h*h - h*h
    eqns[K-1] = y[K-2] - 2*y[K-1] + y[K-1]*h*h + y[K-1]*y[K-1]*h*h - h*h
    
    return eqns

def jac(y):
    j = np.diag(y*2*h*h + h*h - 2)
    j += np.diag(np.ones(K-1), k=1)
    j += np.diag(np.ones(K-1), k=-1)

    print('condition number: ', numpy.linalg.cond(j))
    return j

x = -1 + np.arange(K+2)*h
y0 = numpy.full(K+2, -1)
#y0 = -(x+1)*(x-1)
y02 = y0[1:-1]
infodict = opt.fsolve(func, y02, fprime=jac, full_output=True)
y = infodict[0]
# estimated jacobian condition number
print(numpy.linalg.cond(infodict[1]['fjac']))
#print('solution: ',  y)
#print('closeness: ',  func(y))

np.append(y, 0)
np.insert(y, 0, 0)
plt.plot(np.arange(len(y)), y)
plt.show()
