import jax.numpy as np
import numpy
import matplotlib.pyplot as plt
import time
import jax.random as random
import jax
import scipy.optimize as opt

K = 30
h = 2/(K+1)
x = -1 + np.arange(K+2)*h

def func(y):
    integral = 0
    for i in range(1, len(x)):
        deltaX = x[i] - x[i-1]
        y1 = y[i] - y[i]*y[i]
        y2 = y[i-1] - y[i-1]*y[i-1]
        integral += (y1 + y2)*deltaX/2

    return (np.multiply(x,x) - 1)/2 - integral

y0 = np.full(shape=len(x), fill_value=0)
y = opt.fsolve(func, y0, full_output=False)

plt.plot(np.arange(len(y)), y)
plt.show()



