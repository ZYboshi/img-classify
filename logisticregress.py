from sklearn.linear_model import LogisticRegression

class LogReg:
    def __init__(self):
        self.model = LogisticRegression(
            multi_class='multinomial',  # 多分类设置：softmax
            solver='saga',  # 适合大数据集的优化算法
            max_iter=100,  # 最大迭代次数
            random_state=42,
        )

    def train(self,X,y):
        self.model.fit(X,y)

    def predict(self,X):
        return self.model.predict(X)

