"""
项目：机器学习实验课程设计
数据集：MNIST
模型：神经网络（network）   决策树（decisiontree）   逻辑回归(logisticregression)  贝叶斯分类器
改进：添加各模型训练时间统计
"""
import time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
# 参数
from config import *
from utils import *
# 模型
from network import Net
from decisiontree import DecisionTree
from logisticregress import LogReg
from bayesian import BayesianClassifier
from FCNnet import FullyConnectedNet
# 指标
from sklearn.metrics import accuracy_score, f1_score

def train_pytorch_model(lr=0.01, optimizer_type='adam'):
    train_losses = []

    """训练PyTorch神经网络模型"""
    start_time = time.time()
    model = Net().to(Device)
    criterion = nn.CrossEntropyLoss()
    if optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    # 训练阶段计时
    train_start = time.time()
    for epoch in range(Epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(Device), target.to(Device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"\nEpoch[{epoch}/{Epochs}]\tLoss: {loss.item():.6f}\n")
        train_losses.append(loss.item())

    train_end = time.time()
    train_duration = train_end - train_start

    # 测试阶段计时
    test_start = time.time()
    model.eval()
    y_pred, y_true = [], []
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(Device), target.to(Device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        y_pred.extend(pred.cpu().numpy().flatten())
        y_true.extend(target.cpu().numpy().flatten())
    test_end = time.time()
    test_duration = test_end - test_start

    total_time = time.time() - start_time
    print(f"\n=== 优化器: {optimizer_type.upper()}, 学习率: {lr} ===")
    print("\n******** 卷积神经网络 ********")
    print(f"训练时间: {train_duration:.2f}秒 | 测试时间: {test_duration:.2f}秒 | 总耗时: {total_time:.2f}秒")
    evaluate_classifier(y_true, y_pred)

    return train_losses  # 返回损失列表供外部绘图


def train_decision_tree():
    """训练决策树模型"""
    start_time = time.time()

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # 训练计时
    train_start = time.time()
    model = DecisionTree(max_depth=max_depth)
    model.train(X_train_flat, y_train)
    train_end = time.time()
    train_duration = train_end - train_start

    # 测试计时
    test_start = time.time()
    y_pred = model.predict(X_test_flat)
    test_end = time.time()
    test_duration = test_end - test_start

    total_time = time.time() - start_time

    print("\n******** 决策树 ********")
    print(f"训练时间: {train_duration:.2f}秒 | 测试时间: {test_duration:.2f}秒 | 总耗时: {total_time:.2f}秒")
    evaluate_classifier(y_test, y_pred)

def train_LogisticRegression():
    """训练逻辑回归模型"""
    start_time = time.time()

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # 训练计时
    train_start = time.time()
    model = LogReg()
    model.train(X_train_flat, y_train)
    train_end = time.time()
    train_duration = train_end - train_start

    # 测试计时
    test_start = time.time()
    y_pred = model.predict(X_test_flat)
    test_end = time.time()
    test_duration = test_end - test_start

    total_time = time.time() - start_time

    print("\n******** 逻辑回归 ********")
    print(f"训练时间: {train_duration:.2f}秒 | 测试时间: {test_duration:.2f}秒 | 总耗时: {total_time:.2f}秒")
    evaluate_classifier(y_test, y_pred)

def train_BayesianClassifier():
    """训练贝叶斯分类器"""
    start_time = time.time()

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # 训练计时
    train_start = time.time()
    model = BayesianClassifier()
    model.train(X_train_flat, y_train)
    train_end = time.time()
    train_duration = train_end - train_start

    # 测试计时
    test_start = time.time()
    y_pred = model.predict(X_test_flat)
    test_end = time.time()
    test_duration = test_end - test_start

    total_time = time.time() - start_time

    print("\n******** 贝叶斯分类器 ********")
    print(f"训练时间: {train_duration:.2f}秒 | 测试时间: {test_duration:.2f}秒 | 总耗时: {total_time:.2f}秒")
    evaluate_classifier(y_test, y_pred)

def train_fully_connected():
    """训练全连接神经网络（单层）模型"""
    start_time = time.time()

    model = FullyConnectedNet().to(Device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # 训练阶段计时
    train_start = time.time()
    for epoch in range(Epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(Device), target.to(Device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()



    train_end = time.time()
    train_duration = train_end - train_start

    # 测试阶段计时
    test_start = time.time()
    model.eval()
    y_pred, y_true = [], []
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(Device), target.to(Device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        y_pred.extend(pred.cpu().numpy().flatten())
        y_true.extend(target.cpu().numpy().flatten())
    test_end = time.time()
    test_duration = test_end - test_start

    total_time = time.time() - start_time

    print("\n******** 全连接神经网络（单层） ********")
    print(f"训练时间: {train_duration:.2f}秒 | 测试时间: {test_duration:.2f}秒 | 总耗时: {total_time:.2f}秒")
    evaluate_classifier(y_true, y_pred)

if __name__ == "__main__":
    # 数据加载计时
    data_start = time.time()
    train_loader, test_loader = get_dataloader("MNIST")
    X_train, y_train = get_numpy_data(train_loader)
    X_test, y_test = get_numpy_data(test_loader)
    data_duration = time.time() - data_start
    print(f"数据加载耗时: {data_duration:.2f}秒\n")

    # 根据配置选择模型
    if Model_Type == "pytorch":
        #学习率
        # losses_data = []
        # losses_data.append(('SGD (lr=0.1)', train_pytorch_model(lr=0.1, optimizer_type='sgd')))
        # losses_data.append(('SGD (lr=0.01)', train_pytorch_model(lr=0.01, optimizer_type='sgd')))
        # losses_data.append(('SGD (lr=0.001)', train_pytorch_model(lr=0.001, optimizer_type='sgd')))
        # # 统一绘制所有曲线
        # plt.figure(figsize=(10, 5))
        # for label, losses in losses_data:
        #     plt.plot(losses, label=label)
        #
        # plt.title('Training Loss Curve Comparison')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.grid(True)
        # plt.xticks(np.arange(0, Epochs, step=1))  # 整数刻度
        # plt.xlim(0, Epochs - 1)  # 严格匹配数据点
        # plt.show()

        # 优化器对比分析
        losses_data = []
        lr = 0.001  # 固定学习率

        # 测试不同优化器
        losses_data.append(('SGD', train_pytorch_model(lr=lr, optimizer_type='sgd')))
        losses_data.append(('Adam', train_pytorch_model(lr=lr, optimizer_type='adam')))
        losses_data.append(('RMSprop', train_pytorch_model(lr=lr, optimizer_type='rmsprop')))


        # 统一绘制所有曲线
        plt.figure(figsize=(12, 6))
        for label, losses in losses_data:
            plt.plot(losses, label=label)

        plt.title(f'Optimizer Comparison (lr={lr})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.xticks(np.arange(0, Epochs, step=1))
        plt.xlim(0, Epochs - 1)
        plt.show()

    elif Model_Type == "decision_tree":
        train_decision_tree()
    elif Model_Type == "LogisticRegression":
        train_LogisticRegression()
    elif Model_Type == "Bayesian":
        train_BayesianClassifier()
    elif Model_Type == "FCN":
        train_fully_connected()
    else:
        print("\n========== 运行所有模型 ==========")
        # 全模型运行模式
        full_start = time.time()
        print("\n>>>>>> 训练卷积神经网络 <<<<<<")
        train_pytorch_model()
        print("\n>>>>>> 训练全连接网络 <<<<<<")
        train_fully_connected()
        print("\n>>>>>> 训练决策树 <<<<<<")
        train_decision_tree()

        print("\n>>>>>> 训练逻辑回归 <<<<<<")
        train_LogisticRegression()

        print("\n>>>>>> 训练贝叶斯分类器 <<<<<<")
        train_BayesianClassifier()

        full_duration = time.time() - full_start
        print(f"\n所有模型总运行时间: {full_duration:.2f}秒")