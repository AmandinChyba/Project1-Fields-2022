import jax
import jax.numpy as jnp
import numpy as np
import time
import matplotlib.pyplot as plt

def computeW(n, m, dt, dx):
    # part 1
    key = jax.random.PRNGKey(int(time.time()))
    X = jax.random.choice(key, jnp.array([1,-1]), shape=(m,n))

    W = jnp.zeros(shape=(m,1))
    W = jnp.append(W, X, axis=1)
    W = jnp.cumsum(W, axis=1)

    # part 2a
    a = -50
    b = 50
    
    rows, cols = jnp.where(jnp.logical_or(W >= b,W <= a))

    values, indices = jnp.unique(rows, return_index=True)
   
    result = jnp.take(cols, indices)
    R = W[jnp.arange(len(W)), result]
    aCount = len(R[R < 0])
    bCount = len(R) - aCount
    print("prob of a out: " + str(aCount/len(R)))
    print("prob of b out: " + str(bCount/len(R)))
    
    result = result
    result = jnp.mean(result)

    return result

dt = 1
dx = 1
totalSteps = 100000
totalRuns = 1000

result = computeW(totalSteps, totalRuns, dt, dx)
print("approximation of expected value: " + str(result))

#plt.plot(jnp.arange(0,len(W)*dt, step=dt), W)
#plt.xlabel("time")
#plt.ylabel("W(t)")
#plt.show()


