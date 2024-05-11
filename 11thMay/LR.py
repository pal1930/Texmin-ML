import numpy as np

class LR:
    def __init__(self, N):
        self.theta = np.zeros(N)

    def train(self, X, Y):
        np_X = np.array(X)
        np_Y = np.array(Y)
        X_transpose = np_X.T
        X_transpose_X = np.dot(X_transpose, np_X)
        X_transpose_Y = np.dot(X_transpose, np_Y)

        try:
            self.theta = np.dot(np.linalg.inv(X_transpose_X), X_transpose_Y)
        
        except np.linalg.LinAlgError:
            return None

    def predict(self, X):
        predictions = np.dot(X, self.theta)
        return predictions

obj = LR(2)
obj.train([[1, 1], [2, 3]], [4, 8])
X_test = np.array([7,8])
predictions = obj.predict(X_test)
print("Predictions : " , predictions)