import jax.random as random
import jax.numpy as jnp
import jax
import numpy as np
import time
from tqdm import tqdm

print(jax.default_backend())
def solve_at(g, k, r, theta, t, key):
 
  keys = random.split(key, 2);

  x = random.uniform(keys[0], shape=(1,t)) * 2 * np.pi
  f = k(r, theta - x)
  
  A1 = jnp.tile(jnp.transpose(f), (1,t))
  A2 = jnp.tile(1/f, (t,1))
  
  A = jnp.minimum(1, jnp.tril(jnp.multiply(A1, A2), k=-1))
  u = jnp.tril(random.uniform(keys[1], shape=(t,t)), k=-1)
  check = jnp.subtract(A, u)
    
  col = 0
  y = np.empty(t)
  for i in tqdm(range(t)):
      if check[i+1,col] > 0:
          y[i] = x[1,col]
          col += 1
      else:
          y[i] = x[1,col]
  
  return y, jnp.mean(g(jnp.asarray(y)))
  

def poisson_kernel(r, theta):
  return (1 - jnp.power(r, 2))/(1 - 2 * r * jnp.cos(theta) + jnp.power(r, 2))

t = 1000
r = 0.5
theta = np.pi / 2
key = random.PRNGKey(0)
samples, ans = solve_at(jnp.sin, poisson_kernel, r, theta, t, key)
print(ans)



