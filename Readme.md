# 机器学习实验课程设计
> 西南交通大学  2022116481  张彦博
>
>MNIST手写数字识别数据集在不同模型上的比较

本项目实现了多种机器学习模型在MNIST手写数字识别任务上的训练和预测，包括：

`卷积神经网络 (CNN)`
`全连接神经网络 (FCN)`
`决策树`
`逻辑回归`
`贝叶斯分类器`

## 模型训练准备
1. 数据准备
项目使用MNIST数据集，包含60,000张训练图像和10,000张测试图像。数据加载和预处理通过`utils.py`中的`get_dataloader`函数完成。

2. 训练配置
训练参数在`config.py`中配置，其中模型选择参数`Model_Type`选择为both的时候即为所有模型都执行：  
```python  
# 公共参数
# 决策树参数
# 全连接网络参数

# 模型选择
Model_Type = "both"  # 可选："pytorch", "decision_tree", "LogisticRegression", "Bayesian", "FCN" 或 "both"
```

## 模型训练
>模型训练以及预测都写在同一个函数中，通过调用`main.py`中train_xxx函数即可进行模型训练以及预测

运行`main.py`进行训练+预测
1. 卷积神经网络训练+预测  
CNN模型位置：`network.py`  
`main.py`运行函数:`train_pytorch_model(lr=0.01, optimizer_type='adam')`  

2. 全连接神经网络训练+预测     
单层全连接网络模型位置:`FCNnet.py`  
`main.py`运行函数:`train_fully_connected()`

3. 决策树训练+预测  
决策树模型位置:`decisiontree.py`  
`main.py`运行函数:`train_decision_tree()`

4. 逻辑回归训练+预测  
逻辑回归模型位置:`logisticregress.py`  
`main.py`运行函数:`train_LogisticRegression()`

5. 贝叶斯分类器训练+预测  
贝叶斯分类器模型位置：`train_BayesianClassifier()`  
`main.py`运行函数:`train_BayesianClassifier()`