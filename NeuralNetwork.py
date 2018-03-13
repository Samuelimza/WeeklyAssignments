import numpy as np


class NeuralNetwork:

    def __init__(self, layerN):
        self.m = None
        self.layerN = layerN
        self.w = [2 * np.random.random((layerN[i] + 1, layerN[i + 1])) - 1 for i in range(len(layerN) - 1)]
        self.layer = [None for _ in range(len(layerN))]

    # function to implement forward propagation
    def feedForward(self, X):
        self.layer[0] = self.addBias(X)
        for i in range(1, len(self.layerN) - 1):
            self.layer[i] = self.addBias(self.activate(self.layer[i - 1].dot(self.w[i - 1])))
        self.layer[len(self.layerN) - 1] = self.activate(self.layer[len(self.layerN) - 2].dot(self.w[len(self.layerN) - 2]))

    # function to implement back propagation
    def backProp(self, Y):
        delta = [None for _ in range(len(self.layerN))]
        delta[len(self.layerN) - 1] = Y - self.layer[len(self.layerN) - 1]
        for i in range(len(self.layerN) - 2, 0, -1):
            delta[i] = self.removeRedundantDelta((delta[i + 1].dot(self.w[i].T)) * self.activate(self.layer[i], True))
        for i in range(len(self.layerN) - 1):
            self.w[i] += self.layer[i].T.dot(delta[i + 1]) * (1 / self.m)

    # function to train the NN on passed data
    def train(self, X, Y, iterations):
        self.m = len(X)
        for i in range(iterations):
            self.feedForward(X)
            self.backProp(Y)

    # activation function (sigmoid) that can also calculate g'(z) when flag is set
    @staticmethod
    def activate(layer, flag=False):
        if flag:
            return layer * (1 - layer)
        return 1 / (1 + np.exp(-layer))

    # function to remove delta corresponding to the bias units during back propagation
    @staticmethod
    def removeRedundantDelta(delta):
        newDelta = np.zeros((len(delta), len(delta[0]) - 1))
        for i in range(len(delta)):
            newDelta[i] = delta[i][:-1].copy()
        return newDelta

    # function to add bias units during forward propagation
    @staticmethod
    def addBias(layer):
        newLayer = np.zeros((len(layer), len(layer[0]) + 1))
        for i in range(len(layer)):
            newLayer[i] = np.append(layer[i], [1])
        return newLayer


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

n = NeuralNetwork((3, 4, 4, 1))
n.train(x, y, 10000)

# print output layer after training
n.feedForward(x)
print(n.layer[3])
