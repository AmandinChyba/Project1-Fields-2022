import jax.random as random
import jax.numpy as jnp
import numpy as np
import time
from tqdm import tqdm


def solve_at(g, k, r, theta, t, key):
 
  keys = random.split(key, 2);

  x = random.uniform(keys[0], shape=(1,t)) * 2 * np.pi
  f = k(r, theta - x)
  
  A1 = jnp.tile(f, (t,1))
  A2 = jnp.tile(jnp.transpose(1/f), (1,t))
  
  A = jnp.minimum(1, jnp.tril(jnp.multiply(A1, A2)))
  diag_elements = jnp.diag_indices_from(A)
  A = A.at[diag_elements].set(0)

  u = jnp.tril(random.uniform(keys[1], shape=(t,t)))
  diag_elements = jnp.diag_indices_from(u)
  u = u.at[diag_elements].set(0)
  
  check = jnp.subtract(A, u)
    
  col = 0
  y = []
  for i in tqdm(range(t)):
      if check[i+1,col] > 0:
          y.append(x[1,col])
          col += 1
      else:
          y.append(x[1,col])
  '''
  # should be faster but it is not
  row = 0
  y = []
  col = 0
  while(row < t and col < t):
    # find how many times a proposal x would be rejected
    rowChange = jnp.where(check[row:,col] > 0)[0]
    if (not rowChange.any()):
      rowChange = t - row
    else:
      rowChange = rowChange[0] + 1

    # update stuff
    row = rowChange + row
    y += [x[1,col]]*rowChange
    col += 1
  '''
  #print(jnp.asarray(y))
  #print(g(jnp.asarray(y)))
  return y, jnp.mean(g(jnp.asarray(y)))
  

def poisson_kernel(r, theta):
  return ((1 - jnp.power(r, 2)) / 
          (1 - 2 * r * jnp.cos(theta) + jnp.power(r, 2)))

t = 1000
r = 0.5
theta = np.pi / 2
key = random.PRNGKey(0)
samples, ans = solve_at(jnp.sin, poisson_kernel, r, theta, t, key)
print(ans)



