import numpy as np
from sklearn.naive_bayes import GaussianNB

class BayesianClassifier:
    def __init__(self):
        self.model = GaussianNB(
            var_smoothing=1e-9  # 防止零方差
        )
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    def predict(self, X_test):
        return self.model.predict(X_test)