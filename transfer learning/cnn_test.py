import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import optim,nn

from model import CNN
from train_test import train, test

get_ipython().run_line_magic('matplotlib', 'inline')

"""トレーニングデータの読み込み"""
# 長さ128(2.56sec)のトレーニングデータが7521件/21人分
acc_x_train_df=pd.read_csv('./UCI HAR Dataset/train/Inertial Signals/total_acc_x_train.txt', header=None, sep='\s+', na_values='na')
acc_y_train_df=pd.read_csv('./UCI HAR Dataset/train/Inertial Signals/total_acc_y_train.txt', header=None, sep='\s+', na_values='na')
acc_z_train_df=pd.read_csv('./UCI HAR Dataset/train/Inertial Signals/total_acc_z_train.txt', header=None, sep='\s+', na_values='na')

# Dataframeからarrayへ変換
acc_x_train=acc_x_train_df.values
acc_y_train=acc_y_train_df.values
acc_z_train=acc_z_train_df.values

# 3軸を横に並べる
X_train=np.c_[acc_x_train, acc_y_train, acc_z_train]

# 正解ラベル
    # 1 WALKING
    # 2 WALKING_UPSTAIRS
    # 3 WALKING_DOWNSTAIRS
    # 4 SITTING
    # 5 STANDING
    # 6 LAYING
train_label_df=pd.read_csv('./UCI HAR Dataset/train/Inertial Signals/y_train.txt', header=None, sep='\s+', na_values='na')
y_train=train_label_df.values.reshape(-1)
le = LabelEncoder()
y_train = le.fit_transform(y_train)

# 正解ラベルをone hot表現にする
#ohe = OneHotEncoder()
#y_train = ohe.fit_transform(train_label).A 

"""トレーニングデータの読み込み"""
# 長さ128(2.56sec)のトレーニングデータが7521件/21人分
acc_x_test_df=pd.read_csv('./UCI HAR Dataset/test/Inertial Signals/total_acc_x_test.txt', header=None, sep='\s+', na_values='na')
acc_y_test_df=pd.read_csv('./UCI HAR Dataset/test/Inertial Signals/total_acc_y_test.txt', header=None, sep='\s+', na_values='na')
acc_z_test_df=pd.read_csv('./UCI HAR Dataset/test/Inertial Signals/total_acc_z_test.txt', header=None, sep='\s+', na_values='na')

# Dataframeからarrayへ変換
acc_x_test=acc_x_test_df.values
acc_y_test=acc_y_test_df.values
acc_z_test=acc_z_test_df.values

# 3軸を横に並べる
X_test=np.c_[acc_x_test, acc_y_test, acc_z_test]

# 正解ラベル
    # 1 WALKING
    # 2 WALKING_UPSTAIRS
    # 3 WALKING_DOWNSTAIRS
    # 4 SITTING
    # 5 STANDING
    # 6 LAYING
test_label_df=pd.read_csv('./UCI HAR Dataset/test/Inertial Signals/y_test.txt', header=None, sep='\s+', na_values='na')
y_test=test_label_df.values.reshape(-1)
le = LabelEncoder()
y_test = le.fit_transform(y_test)
 
# 1. GPUの設定（PyTorchでは明示的に指定する必要がある）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 2. ハイパーパラメータの設定（最低限の設定）
batch_size = 100
num_classes = 6
epochs = 10

# 5-2. データのフォーマットを変換：PyTorchでの形式 = [画像数，チャネル数，高さ，幅]
X_train = X_train.reshape(-1, 1, 3, 128)
X_test = X_test.reshape(-1, 1, 3 ,128)

# 5-3. PyTorchのテンソルに変換
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)
 
# 5-4. 入力(x)とラベル(y)を組み合わせて最終的なデータを作成
ds_train = TensorDataset(X_train, y_train)
ds_test = TensorDataset(X_test, y_test)
 
# 5-5. DataLoaderを作成
loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
 
# 6. モデル作成
model = CNN(num_classes=num_classes).to(device)
print(model) # ネットワークの詳細を確認用に表示
 
# 7. 損失関数を定義
loss_fn = nn.CrossEntropyLoss()
 
# 8. 最適化手法を定義（ここでは例としてAdamを選択）
optimizer = optim.Adam(model.parameters(), lr=0.01)
 
# 9. 学習（エポック終了時点ごとにテスト用データで評価）
print('Begin train')
for epoch in range(1, epochs+1):
    train(loader_train, model, optimizer, loss_fn, device, epochs, epoch)
    test(loader_test, model, device)


"""　転移学習(最終全結合層のみtrain) """
# すべての層のパラメータを固定
for param in model.parameters():
    param.requires_grad = False
# 新しくできた層はデフォルトで，requires_grad=True
num_ftrs = model.fc[3].in_features # 最後の全結合層のユニット数
model.fc[3] = nn.Linear(num_ftrs, 6)
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc[3].parameters(), lr=0.01)

print('Begin train')
for epoch in range(1, epochs+1):
    train(loader_train, model, optimizer, loss_fn, device, epochs, epoch)
    test(loader_test, model, device)


"""　転移学習(fine tuning) """
# 最後の全結合層をリセット
num_ftrs = model.fc[3].in_features # 最後の全結合層のユニット数
model.fc[3] = nn.Linear(num_ftrs, 6)
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print('Begin train')
for epoch in range(1, epochs+1):
    train(loader_train, model, optimizer, loss_fn, device, epochs, epoch)
    test(loader_test, model, device)
