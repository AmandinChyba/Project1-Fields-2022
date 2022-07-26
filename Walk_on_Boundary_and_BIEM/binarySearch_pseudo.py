import numpy as np

# x0,y0 is the starting point
x0 = 0
y0 = 0

# x1, y1 is just some far away point outside the boundary
x1 = 10
y1 = 10

epochs = 100
for i in range(epochs):
    # update the middle point
    xnew = x1 - x0
    ynew = y1 - y0

    # 'boundary' is actually the signed distance function
    if boundary(xnew, ynew) > 0:
        x1 = xnew
        y1 = ynew
    else:
        x0 = xnew
        y0 = xnew

# the root is at this point
xnew = x1 - x0
ynew = y1 - y0



