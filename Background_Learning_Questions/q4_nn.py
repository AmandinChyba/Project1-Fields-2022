import jax
import jax.random as random
import jax.numpy as np
import pandas as pd
import time

def f(x):
    return np.exp(-10*np.abs(x))

def generateData(size, fileName):
    '''Generate new training data using the function f'''
    key = random.PRNGKey(int(time.time()))

    xData = random.uniform(key, shape=(size,1), minval=-1, maxval=1)
    yData = f(xData)
    
    data = np.concatenate((xData, yData), axis=1)
    df = pd.DataFrame(data=data, columns=['x', 'y'])

    df.to_pickle('./pickle_files/' + fileName + '.pkl')

#generateData(10000, 'q4_dataSet')

def ReLU(x):
    return np.maximum(0,x)

def initializeParam(layers):
    '''Initilaze the weights and bias of each neuron in the network's layers'''
    key = random.PRNGKey(int(time.time()))
    
    # create the first layer
    weights = np.array(random.uniform(key, shape=(layers[0],1), minval=-1, maxval=1))
    bias = np.array(random.uniform(key, shape=(layers[0],1), minval=-1, maxval=1))
    
    # create a list of all the weights and biases as a tuple for each layer
    param = [(weights,bias)]

    # go through each layer after the first
    for layer in range(len(layers)-1):
        # generate weights based on the number of neurons in previous layer
        weights = random.uniform(key, shape=(layers[layer+1],layers[layer]), minval=-1, maxval=1)
        bias = np.array(random.uniform(key, shape=(layers[layer+1],1), minval=-1, maxval=1))
        
        param.append((weights,bias))

    print(param)

    


# to read in the dataset -> df = df.read_pickle('./pickle_files/' + fileName + '.pkl')
layers = np.array([3, 5, 1])
initializeParam(layers)










