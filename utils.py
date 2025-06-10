from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
#评估工具
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report

from config import *

def get_dataloader(datasets_name):
    
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为Tensor
        transforms.Normalize((0.5,), (0.5,))  # 归一化处理
    ])

    if datasets_name=="MNIST":
        
        train_dataset = datasets.MNIST(
            root='./data',  # 数据保存的路径
            train=True,  # 表示这是训练集
            transform=transform,  # 应用的转换
            download=True  # 如果数据不存在，则从互联网下载
        )

        test_dataset = datasets.MNIST(
            root='./data',
            train=False,  # 表示这是测试集
            transform=transform,
            download=True
        )

        train_dataloader=DataLoader(train_dataset,batch_size=BatchSize,shuffle=True)
        test_dataset=DataLoader(test_dataset,batch_size=BatchSize,shuffle=False)
    
    else:
        raise Exception(f"没用写这个：{datasets_name} 数据集的获取")


    return train_dataloader,test_dataset

#将tensor->numpy
def get_numpy_data(dataloader):
    """将PyTorch DataLoader转换为numpy数组（保持图像二维结构）"""
    X_list, y_list = [], []
    for images, labels in dataloader:
        # 保持图像二维结构 (batch_size, 1, 28, 28)
        X_list.append(images.numpy())
        y_list.append(labels.numpy())

    X = np.concatenate(X_list, axis=0)  # shape: (N, 1, 28, 28)
    y = np.concatenate(y_list, axis=0)  # shape: (N,)

    # 去掉通道维度 (N, 1, 28, 28) -> (N, 28, 28)
    X = X.squeeze(axis=1)
    return X, y


def evaluate_classifier(y_true, y_pred):
    """
    通用分类器评估函数（同时输出macro和micro指标）
    参数:
        y_true: 真实标签（形状 [n_samples]）
        y_pred: 预测标签（形状 [n_samples]）
    返回:
        dict: 包含所有指标的字典
    """
    # 基础指标（accuracy与average无关）
    accuracy = accuracy_score(y_true, y_pred)

    # 计算macro和micro指标
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    precision_micro = precision_score(y_true, y_pred, average='micro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    f1_micro = f1_score(y_true, y_pred, average='micro')

    # 打印报告
    print("\n========== 分类器评估报告 ==========")
    print(f"准确率(Accuracy): {accuracy:.4f}")

    print("\n=== Macro平均（各类别平等权重）===")
    print(f"精确率(Precision): {precision_macro:.4f}")
    print(f"召回率(Recall): {recall_macro:.4f}")
    print(f"F1值: {f1_macro:.4f}")

    print("\n=== Micro平均（全局样本权重）===")
    print(f"精确率(Precision): {precision_micro:.4f}")
    print(f"召回率(Recall): {recall_micro:.4f}")
    print(f"F1值: {f1_micro:.4f}")

    # 打印详细分类报告
    print("\n详细分类报告（按类别）:")
    print(classification_report(y_true, y_pred, digits=4))

    return {
        'accuracy': accuracy,
        'macro': {
            'precision': precision_macro,
            'recall': recall_macro,
            'f1': f1_macro
        },
        'micro': {
            'precision': precision_micro,
            'recall': recall_micro,
            'f1': f1_micro
        }
    }