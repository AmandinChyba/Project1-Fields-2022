import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import time

def g(theta): # boundary condition
    return np.sin(theta)

def A(r, theta, x, xprime): # poisson kernel on unit disk
    lol = (1 - 2*r*np.cos(theta-x) + np.power(r,2))/(1 - 2*r*np.cos(theta-xprime) + np.power(r,2))
    return np.minimum(1, lol)

def graph3d():
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    theta = np.linspace(0, 2*np.pi, 1000)
    x = np.cos(theta)
    y = np.sin(theta)
    z = g(theta)

    x = np.append(x, 0)
    y = np.append(y, 0)
    z = np.append(z, 0)

    ax.scatter3D(x, y, z);
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    
    plt.grid(True)
    plt.show()

def graphAcceptance():
    fig = plt.figure()
    
    r = 0.5
    theta = np.pi/2
    phi = np.linspace(0, 2*np.pi, 1000)
    current = 1.47
    x = np.cos(phi)
    y = np.sin(phi)
    color = A(r,theta,current,phi)
    
    plt.scatter(x, y, c=color, edgecolor='none');
    plt.scatter(np.cos(current), np.sin(current), s=100,
                facecolors='none', edgecolor='red', linewidth=2)
    plt.xlabel('x axis')
    plt.ylabel('y axis')

    plt.colorbar(label='acceptance probability', orientation='vertical')
    plt.grid(True)
    plt.show()


#graph3d()
graphAcceptance()




