import jax
from jax.example_libraries import optimizers as jax_opt
import jax.random as random
import jax.numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

def f(x):
    '''Function we want to train our ANN for'''
    return np.exp(-10*np.abs(x))
    #return x

def generateData(size, fileName):
    '''Generate new training data using the function f'''
    key = random.PRNGKey(int(time.time()))

    xData = random.uniform(key, shape=(size,1), minval=-1, maxval=1)
    yData = f(xData)
    
    data = np.concatenate((xData, yData), axis=1)
    df = pd.DataFrame(data=data, columns=['x', 'y'])

    df.to_pickle('./pickle_files/' + fileName + '.pkl')

def initializeParam(layers):
    '''Initilaze the weights and bias of each neuron in the network's layers'''
    key = random.PRNGKey(int(time.time()))
    keys = random.split(key) 
    
    # create the first layer
    weights = np.array(random.uniform(keys[0], shape=(layers[0],1), minval=-1, maxval=1))
    bias = np.array(random.uniform(keys[1], shape=(layers[0],1), minval=-1, maxval=1))
    
    # create a list of all the weights and biases as a tuple for each layer
    params = [[weights,bias]]
    #allWeights = [weights]
    #allBias = [bias]

    # go through each layer after the first
    for layer in range(len(layers)-1):
        # generate weights based on the number of neurons in previous layer
        weights = random.uniform(key, shape=(layers[layer+1],layers[layer]), minval=0, maxval=1)
        bias = np.array(random.uniform(key, shape=(layers[layer+1],1), minval=-1, maxval=1))
        
        #allWeights.append(weights)
        #allBias.append(bias)
        params.append([weights,bias])
    
    return params
    #return allWeights, allBias

def ReLU(x):
    return np.maximum(0,x)

def sigmoid(x):
    return 0.5 * (np.tanh(x / 2) + 1)

def forwardPass(params, x):
    '''Go through the network once'''
    values = np.array(x)
    
    # go through each layer
    for i in range(len(params)):
        #values = sigmoid(np.dot(W[i], values) + b[i])
        values = np.dot(params[i][0], values) + params[i][1]
        if (i != len(params)-1):
            #values = np.apply_along_axis(sigmoid, 0, values)
            #values = np.apply_along_axis(ReLU, 1, values)
            values = ReLU(values)
    
    # give back the predicted value
    return values[0][0]
    

def loss(pred, actual):
    '''Calculate the residual with a loss function'''
    return np.power(pred-actual, 2)

def predict(params, x, y):
    '''Apply the network to the input data and compute the loss'''
    # scuffed batching
    totalRes = []
    for i in range(len(x)):
        totalRes.append(loss(forwardPass(params, x[i]), y[i]))
    
    return np.mean(np.array(totalRes))
    #pred = forwardPass(W, b, x)
    #res = loss(pred, y)
    #return res

def backwardPass(W_grads, b_grads, W, b, lr):
    '''Update weights and bias so that the network learns'''
    for i in range(len(W)):
        W[i] = W[i] - lr * W_grads[i]
        b[i] = b[i] - lr * b_grads[i]
    
fileName = 'q4_dataSet'

# get data
#generateData(100000, fileName)
df = pd.read_pickle('./pickle_files/' + fileName + '.pkl')
xTrain = df['x'].tolist()
yTrain = df['y'].tolist()

# create ANN
layers = np.array([50,10,1])
#W, b = initializeParam(layers)
params = initializeParam(layers)

# train the ANN
lr = 0.001
batchSize = 1
epochs = 90000
opt_init, opt_update, get_params = jax_opt.adam(lr)
opt_state = opt_init(params)
for i in range(0, len(xTrain), batchSize):
    # forward propagate
    #W_grads, b_grads = jax.grad(predict, (0,1))(W, b, np.array(xTrain[i:i+batchSize]), np.array(yTrain[i:i+batchSize]))
    W_grads = jax.grad(predict, 0)(get_params(opt_state), np.array(xTrain[i:i+batchSize]), np.array(yTrain[i:i+batchSize]))
    opt_state = opt_update(0, W_grads, opt_state)
    
    # backward propagate
    #backwardPass(W_grads, b_grads, W, b, lr)
    
    # record loss
    if (i % 1000 == 0):
        #print('loss: ', predict(W, b, np.array(xTrain[i]), np.array(yTrain[i])))
        print('guess: ' + str(forwardPass(get_params(opt_state), np.array(0))))
    
    # stopping point
    if (i==epochs):
        break

# test the ANN
yPred = []
xPred = []
for i in range(len(xTrain)):
    if (i>epochs and i < epochs+100):
        yPred.append(forwardPass(get_params(opt_state), xTrain[i]))
        xPred.append(xTrain[i])

# plot the results
plt.scatter(xTrain, yTrain)
plt.scatter(xPred, yPred)
plt.show()






