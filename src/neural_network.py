import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.W1, self.b1, self.W2, self.b2 = self.init_params()

    def init_params(self):
        W1 = np.random.rand(10, 784) - 0.5
        b1 = np.random.rand(10, 1) - 0.5
        W2 = np.random.rand(10, 10) - 0.5
        b2 = np.random.rand(10, 1) - 0.5
        return W1, b1, W2, b2

    def ReLU(self, Z):
        return np.maximum(Z, 0)

    def softmax(self, Z):
        A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        return A

    def forward_prop(self, X):
        self.Z1 = self.W1.dot(X) + self.b1
        self.A1 = self.ReLU(self.Z1)
        self.Z2 = self.W2.dot(self.A1) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.Z1, self.A1, self.Z2, self.A2

    def ReLU_deriv(self, Z):
        return Z > 0

    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def backward_prop(self, X, Y, m):
        one_hot_Y = self.one_hot(Y)
        dZ2 = self.A2 - one_hot_Y
        dW2 = 1 / m * dZ2.dot(self.A1.T)
        db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = self.W2.T.dot(dZ2) * self.ReLU_deriv(self.Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
        return dW1, db1, dW2, db2

    def update_params(self, dW1, db1, dW2, db2, alpha):
        self.W1 -= alpha * dW1
        self.b1 -= alpha * db1
        self.W2 -= alpha * dW2
        self.b2 -= alpha * db2

    def get_predictions(self):
        return np.argmax(self.A2, 0)

    def get_accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.size

    def gradient_descent(self, X, Y, alpha, iterations):
        m = X.shape[1]
        for i in range(iterations):
            self.forward_prop(X)
            dW1, db1, dW2, db2 = self.backward_prop(X, Y, m)
            self.update_params(dW1, db1, dW2, db2, alpha)
            if i % 10 == 0:
                print("Iteration:", i)
                predictions = self.get_predictions()
                print("Accuracy:", self.get_accuracy(predictions, Y))

    def make_predictions(self, X):
        _, _, _, A2 = self.forward_prop(X)
        predictions = self.get_predictions()
        return predictions

    def test_prediction(self, X_train, Y_train, index):
        current_image = X_train[:, index, None]
        prediction = self.make_predictions(current_image)
        label = Y_train[index]
        return prediction, label, current_image.reshape((28, 28)) * 255
