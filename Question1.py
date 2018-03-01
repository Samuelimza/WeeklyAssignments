# ----Fully Vectorized Implementation of Forward and Back Propagation---- #

import numpy as np

# Parameters controlling the no of neurons in each layer
nLayer0 = 3
nLayer1 = 4
nLayer2 = 4
nLayer3 = 1

# input data
x = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
])

# output data
y = np.array([
    [0],
    [1],
    [1],
    [0],
    [1],
    [0],
    [0],
    [1]
])

m = len(y)

np.random.seed(1)

w0 = 2 * np.random.random((nLayer0 + 1, nLayer1)) - 1
w1 = 2 * np.random.random((nLayer1 + 1, nLayer2)) - 1
w2 = 2 * np.random.random((nLayer2 + 1, nLayer3)) - 1


# function to implement forward propagation
def feedForward(X, w0, w1, w2):
    layer0 = X
    layer0 = addBias(layer0)
    layer1 = activate(layer0.dot(w0))
    layer1 = addBias(layer1)
    layer2 = activate(layer1.dot(w1))
    layer2 = addBias(layer2)
    layer3 = activate(layer2.dot(w2))
    return layer3, layer2, layer1, layer0


# function to implement back propagation
def backProp(X, Y, w0, w1, w2):
    layer3, layer2, layer1, layer0 = feedForward(X, w0, w1, w2)
    delta3 = Y - layer3
    delta2 = (delta3.dot(w2.T)) * activate(layer2, True)
    delta2 = removeRedundantDelta(delta2)
    delta1 = (delta2.dot(w1.T)) * activate(layer1, True)
    delta1 = removeRedundantDelta(delta1)
    w0 += layer0.T.dot(delta1) * (1 / m)
    w1 += layer1.T.dot(delta2) * (1 / m)
    w2 += layer2.T.dot(delta3) * (1 / m)
    return w0, w1, w2


# activation function (sigmoid) that can also calculate g'(z) when flag is set
def activate(layer, flag=False):
    if flag:
        return layer * (1 - layer)
    return 1 / (1 + np.exp(-layer))


# function to remove delta corresponding to the bias units during back propagation
def removeRedundantDelta(delta):
    newDelta = np.zeros((len(delta), len(delta[0]) - 1))
    for i in range(len(delta)):
        newDelta[i] = delta[i][:-1].copy()
    # print(newDelta)
    return newDelta


# function to add bias units during forward propagation
def addBias(layer):
    newLayer = np.zeros((len(layer), len(layer[0]) + 1))
    for i in range(len(layer)):
        newLayer[i] = np.append(layer[i], [1])
        # layer[i].append(1)
    # print(newLayer)
    return newLayer


# print output layer before training
l3, _, _, _, = feedForward(x, w0, w1, w2)
print(l3)

for counter in range(10000):
    w0, w1, w2 = backProp(x, y, w0, w1, w2)

# print output layer after training
l3, _, _, _, = feedForward(x, w0, w1, w2)
print(l3)
