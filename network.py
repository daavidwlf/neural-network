import numpy as np
import pandas as pd



### import and convert data
data  =  pd.read_csv('./mnist_train.csv')
data = np.array(data)


## randomize and slice the data
np.random.shuffle(data)
data = data[0:40000]
print(np.shape(data))



### some data parsing
amtData, amtPixel = np.shape(data)
label = data.T[0, :]
pixel = data.T[1:amtData, :]



### weights and biases
def initializeWeights():
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    w2  = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return w1, b1, w2, b2

def initializeWeightsAlt():
    w1 = np.random.normal(size=(10, 784)) * np.sqrt(1./(784))
    b1 = np.random.normal(size=(10, 1)) * np.sqrt(1./10)
    w2 = np.random.normal(size=(10, 10)) * np.sqrt(1./20)
    b2 = np.random.normal(size=(10, 1)) * np.sqrt(1./(784))
    return w1, b1, w2, b2

def updateParams(w1, b1, w2, b2, nw1, nb1, nw2, nb2, rate):
    w1 = w1 - rate * nw1
    b1 = b1 - rate * np.reshape(nb1, (10,1))
    w2 = w2 - rate * nw2
    b2 = b2 - rate * np.reshape(nb1, (10,1))
    return w1, b1, w2, b2



### activiation functions
def reLu(array):
    return np.maximum(array, 0)

def softmax(array):
    maxValues = np.max(array, axis=0)
    array  =  array - maxValues
    expValues = np.exp(array)
    expSum = np.sum(expValues, axis=0)
    result = expValues / expSum
    return result

def softmaxAlt(array):
    array = array - np.max(array, axis=0)
    result = np.exp(array) / np.sum(np.exp(array), axis=0)
    return result

### activiation functions derevatives
def reLuDerivative(array):
    return array > 0



### function to get the optimal a2 vectors
def getActualVector():
    result = np.zeros((10, label.size))
    result[label, np.arange(label.size)] = 1
    return result
    


#### neural network
def forwardProp(pixel, w1, b1, w2, b2):
    u1 = w1.dot(pixel)
    u1 = u1 + b1
    a1 = reLu(u1)
    u2 = w2.dot(a1)
    u2 = u2 + b2
    a2 = softmax(u2)
    return u1, a1, u2, a2

def backwardProp(u1, a1, u2, a2, w1, b1, w2, b2):
    lossLayer2 = a2 - getActualVector()
    nw2 = 1/amtData * lossLayer2.dot(a1.T)
    nb2 =  1/amtData * np.sum(lossLayer2, axis=1, keepdims=True)
    lossLayer1  = w2.dot(lossLayer2) * reLuDerivative(u1)
    nw1 = 1/amtData * lossLayer1.dot(pixel.T)
    nb1 = 1/amtData * np.sum(lossLayer1,  axis=1, keepdims=True)
    return nw1, nb1, nw2, nb2



##calc accuracy
def getAccuracy(estimaed, label):
    #print(estimaed)
    #print(label)
    return np.sum(estimaed == label) / amtData



### gradient descend to reduce costs
def gradientDescend(iterations, rate):
    w1, b1, w2, b2 = initializeWeightsAlt()
    for i in range(iterations+1):
        u1, a1, u2, a2 = forwardProp(pixel, w1, b1, w2, b2)
        nw1, nb1, nw2, nb2 = backwardProp(u1, a1, u2, a2, w1, b1, w2, b2)
        w1, b1, w2, b2 = updateParams(w1, b1, w2, b2, nw1, nb1, nw2, nb2, rate)


        if(i % 20 == 0):
            print("Iteration:", i)
            estimated = np.argmax(a2, axis=0)
            print("Accuracy:",getAccuracy(estimated, label)*100, "%")


###start
gradientDescend(500, 0.0001)
    


###testing and debugging
#result = np.argmax(a2, axis=1)
#testData = np.array([
#    [1, 2, 3, 6],
#    [2, 4, 5, 6],
#    [1, 2, 3, 6]
#])
#softmax(testData)
#forwardProp