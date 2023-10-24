import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 读取CSV文件
data = pd.read_csv('dataset.csv')

# 提取特征和标签
X = data.iloc[:, :2].values  # 前两列作为特征
y = data.iloc[:, -1].values  # 最后一列作为标签

# 划分训练集、验证集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

# 构建子分类器
classifier1 = ExtraTreesClassifier(n_estimators=100, max_depth=5)
classifier2 = RandomForestClassifier(n_estimators=100, max_depth=5)
classifier3 = BaggingClassifier(n_estimators=100, max_samples=0.5)

# 构建投票分类器
voting_classifier = VotingClassifier(estimators=[
    ('extra_trees', classifier1),
    ('random_forest', classifier2),
    ('bagging', classifier3)
], voting='soft')

# 在训练集上拟合模型
voting_classifier.fit(X_train, y_train)

# 在测试集上进行预测
y_test_pred = voting_classifier.predict(X_test)

# 计算测试集的准确率
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", test_accuracy)

# 在验证集上进行预测
y_val_pred = voting_classifier.predict(X_val)

# 计算验证集的准确率
val_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", val_accuracy)



