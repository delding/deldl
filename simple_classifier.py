#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

N = 100  # number of points per class
D = 2  # dimensionality
K = 3  # number of classes
X = np.zeros((N*K, D))  # data matrix
y = np.zeros(N*K, dtype='uint8')  # class labels
for j in range(K):
    ix = range(N*j, N*(j+1))
    r = np.linspace(0.0, 1, N)  # radius
    t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2  # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()


class FC(object):
    def __init__(self, shape):
        W = 0.01 * np.random.randn(*shape)
        self.W = np.append(W, np.zeros((1, W.shape[1])), axis=0)  # add bias param

    def forward(self, inputs):
        self.X = np.append(inputs, np.ones((inputs.shape[0], 1)), axis=1)  # add bias input
        return self.X.dot(self.W)

    def backward(self, doutputs):
        dX = doutputs.dot(self.W.T)[:, 0:self.W.T.shape[1]-1]  # remove bias input
        dW = self.X.T.dot(doutputs)
        return dX, dW


class RELU(object):
    def forward(self, inputs):
        self.X = inputs
        return np.maximum(0, inputs)

    def backward(self, doutputs):
        doutputs[self.X <= 0] = 0
        return doutputs


class SimpleClassifier(object):
    def __init__(self, X, label, hidden_layer_size=120,
                 learning_rate=1e-0, reg=1e-3):
        self.X = X
        self.label = label
        self.learning_rate = learning_rate
        self.reg = reg
        self.fc1 = FC((self.X.shape[1], hidden_layer_size))
        self.relu = RELU()
        self.fc2 = FC((hidden_layer_size, K))

    def train(self, epoch=2000):
        for i in range(epoch):
            h1 = self.fc1.forward(self.X)
            h1 = self.relu.forward(h1)
            logits = self.fc2.forward(h1)
            exps = np.exp(logits)
            probs = exps / np.sum(exps, axis=1, keepdims=True)

            logprobs = np.log(probs)
            predicted = np.argmax(logprobs, axis=1)
            cross_entropies = -1 * (logprobs[range(logprobs.shape[0]), self.label])
            reg_loss = (0.5 * self.reg * np.sum(self.fc1.W * self.fc1.W) +
                        0.5 * self.reg * np.sum(self.fc2.W * self.fc2.W))
            loss = np.sum(cross_entropies) / cross_entropies.shape[0] + reg_loss
            if (i + 1) % 10 == 0:
                print(f"iteration {i + 1}: training loss {loss}, "
                      f"accuracy {np.mean(predicted == y)}")

            # d_cross_entropies / d_logits
            dlogits = probs
            dlogits[range(dlogits.shape[0]), self.label] -= 1
            dlogits /= dlogits.shape[0]

            dX2, dW2 = self.fc2.backward(dlogits)
            dW2 += self.reg * self.fc2.W
            dX2 = self.relu.backward(dX2)
            _, dW1 = self.fc1.backward(dX2)
            dW1 += self.reg * self.fc1.W
            self.fc1.W -= self.learning_rate * dW1
            self.fc2.W -= self.learning_rate * dW2

    def predict(self, x0, x1):
        sample = np.array([[x0, x1]])
        h1 = self.fc1.forward(sample)
        h1 = self.relu.forward(h1)
        score = self.fc2.forward(h1).flatten()
        klass = np.argmax(score)
        return klass


sc = SimpleClassifier(X, y)
sc.train()

X = []
y = []
x0 = -1.0
while x0 <= 1.0:
    x0 += 0.01
    x1 = -1.0
    while x1 <= 1.0:
        x1 += 0.01
        X.append([x0, x1])
        y.append(sc.predict(x0, x1))
X = np.array(X)
y = np.array(y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()
