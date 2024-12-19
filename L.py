import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# 读取数据集
df_train = pd.read_csv(r"C:\Users\Administrator\PycharmProjects\my_pythonProject\train.csv")
df_test = pd.read_csv(r"C:\Users\Administrator\PycharmProjects\my_pythonProject\testA.csv")

# 初始化数据列表
floats_train = []
y_train_filtered = []

# 处理训练集数据
for signal, label in zip(df_train['heartbeat_signals'], df_train['label']):
    try:
        signal_list = signal.split(',')
        floats_train.append([float(x) for x in signal_list])
        y_train_filtered.append(label)
    except ValueError:
        continue

# 转换为 NumPy 数组
floats_train_array = np.array(floats_train)
y_train_filtered = np.array(y_train_filtered)

# 标准化数据
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(floats_train_array)

# 独热编码标签
encoder = OneHotEncoder(sparse_output=False)
y_train_one_hot = encoder.fit_transform(y_train_filtered.reshape(-1, 1))

# 处理测试集数据
floats_test = []
for signal in df_test['heartbeat_signals']:
    try:
        signal_list = signal.split(',')
        floats_test.append([float(x) for x in signal_list])
    except ValueError:
        continue

floats_test_array = np.array(floats_test)
x_test_scaled = scaler.transform(floats_test_array)

# 划分训练集和验证集
x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
    x_train_scaled, y_train_one_hot, test_size=0.2, random_state=42
)

# 导入 PyTorch 相关库
import torch
import torch.nn as nn
import torch.optim as optim

# 定义 CNN 模型
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * x_train_split.shape[1], 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型
num_classes = 4  # 根据实际情况设置
model = CNNModel(num_classes)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 转换数据为 PyTorch 张量
X_train_tensor = torch.from_numpy(x_train_split).float().unsqueeze(1)
y_train_tensor = torch.from_numpy(y_train_split.argmax(axis=1)).long()

# 训练模型
n_epochs = 200
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
# 打印每个 epoch 的损失
    print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}')
# 预测测试集
X_test_tensor = torch.from_numpy(x_test_scaled).float().unsqueeze(1)
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_probabilities = torch.softmax(test_outputs, dim=1).numpy()

# 生成提交文件
sample_submit = pd.read_csv(r"C:\Users\Administrator\PycharmProjects\my_pythonProject\sample_submit.csv")
sample_submit[['label_0', 'label_1', 'label_2', 'label_3']] = test_probabilities
sample_submit.to_csv('sample_submit.csv', index=False)
print("预测结果已保存到 sample_submit.csv")

from tensorflow.keras.utils import plot_model

# 使用前面定义的模型
model = CNNModel(num_classes=4)

# 绘制模型图
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)