# 导入必要的库
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# ==========================
# 1. 数据加载
# ==========================
# 加载训练集和测试集
train = pd.read_csv('D:/pycharm/kaggle/房价预测/train.csv')
test = pd.read_csv('D:/pycharm/kaggle/房价预测/test.csv')

# 显示前几行数据，检查数据结构
print("训练集数据：")
print(train.head())
print("测试集数据：")
print(test.head())

# ==========================
# 2. 数据预处理
# ==========================
# 删除不需要的列（如 'Id' 列），只删除训练集中的 'Id' 列,inplace=True：指定是否修改原始数据框。
train.drop('Id', axis=1, inplace=True)

# 保留测试集的 'Id' 列，方便后续生成提交文件
test_id = test['Id']  # 保存测试集中的 'Id'

# 删除测试集中的 'Id' 列
test.drop('Id', axis=1, inplace=True)

# 处理缺失值
# 合并训练集和测试集，便于统一处理
full_data = pd.concat([train, test], axis=0, ignore_index=True)
# ignore_index=True：重新生成索引。

# 填充数值型数据的缺失值，使用中位数
num_cols = full_data.select_dtypes(include=['int64', 'float64']).columns
full_data[num_cols] = full_data[num_cols].fillna(full_data[num_cols].median())

# 填充分类数据的缺失值，使用众数
cat_cols = full_data.select_dtypes(include=['object']).columns
full_data[cat_cols] = full_data[cat_cols].fillna(full_data[cat_cols].mode().iloc[0])
# mode() 是一个统计方法，返回每一列中最常见的值（众数）。
# iloc[0]：选择众数中的第一个值（如果有多个众数，选择第一个）。

# ==========================
# 3. 特征编码
# ==========================
# 使用独热编码转换分类特征
full_data = pd.get_dummies(full_data, drop_first=True)
# drop_first=True：删除每个分类变量的第一个类别列，防止多重共线性问题。
print("数据处理后的形状：", full_data.shape)

# ==========================
# 4. 划分数据集
# ==========================
# 重新拆分训练集和测试集
X = full_data[:len(train)]  # 训练集特征
X_test = full_data[len(train):]  # 测试集特征
y = train['SalePrice']  # 训练集目标值

# ==========================
# 5. 模型训练
# ==========================
# 创建随机森林回归模型
model = RandomForestRegressor(
    n_estimators=400,  # 使用400棵树
    random_state=42,  # 设置随机种子
    n_jobs=-1  # 使用所有CPU核
)

# 训练模型
model.fit(X, y)

# ==========================
# 6. 进行预测
# ==========================
# 使用训练好的模型进行预测
predictions = model.predict(X_test)

# ==========================
# 7. 创建提交文件
# ==========================
# 创建提交文件
submission = pd.DataFrame({
    "Id": test_id,  # 使用之前保存的测试集 'Id'
    "SalePrice": predictions  # 预测的房价
})

# 保存提交文件
submission.to_csv("submission.csv", index=False)

print("提交文件已保存：submission.csv")
