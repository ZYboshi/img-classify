# decision_tree.py
import torch
import numpy as np
from sklearn.tree import DecisionTreeClassifier  # 使用scikit-learn的决策树

class DecisionTree:
    def __init__(self, max_depth=None):
        self.model = DecisionTreeClassifier(max_depth=max_depth,criterion='entropy')

    def train(self, X_train, y_train):
        # 输入需转为numpy数组（决策树不直接处理张量）
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)