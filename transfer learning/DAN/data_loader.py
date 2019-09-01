import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader

def data_preparation(root_path, directory, mode):
    file_path_data = glob.glob(root_path + directory + '/total_acc_*.txt')
    acc_x=pd.read_csv(file_path_data[0], header=None, sep='\s+', na_values='na').values
    acc_y=pd.read_csv(file_path_data[1], header=None, sep='\s+', na_values='na').values
    acc_z=pd.read_csv(file_path_data[2], header=None, sep='\s+', na_values='na').values
    X=np.c_[acc_x, acc_y, acc_z] # 3軸を横に並べる
    
    file_path_label = glob.glob(root_path + directory + '/y_*.txt')
    le = LabelEncoder()
    y=pd.read_csv(file_path_label[0], header=None, sep='\s+', na_values='na').values.reshape(-1)
    y = le.fit_transform(y)
    
    return X, y

def load_training(root_path, directory, batch_size=256):
    X_train, y_train = data_preparation(root_path, directory, mode='train')
    X_train = X_train.reshape(-1, 1, 3, 128) # フォーマット変換：[画像数，チャネル数，高さ，幅]
    X_train = torch.Tensor(X_train)
    y_train = torch.LongTensor(y_train)
    
    ds = TensorDataset(X_train, y_train)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return loader
    
def load_testing(root_path, directory):
    X_test, y_test = data_preparation(root_path, directory, mode='test')
    X_test = X_test.reshape(-1, 1, 3, 128) # フォーマット変換：[画像数，チャネル数，高さ，幅]
    X_test = torch.Tensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    ds = TensorDataset(X_test, y_test)
    loader = DataLoader(ds, batch_size=len(ds), shuffle=True)
    
    return loader

if __name__=='__main__':
    source_loader = load_training(root_path='../UCI HAR Dataset', directory='/train/Inertial Signals', batch_size=256)
    target_train_loader = load_training(root_path='../UCI HAR Dataset', directory='/test/Inertial Signals', batch_size=256)
    target_test_loader = load_testing(root_path='../UCI HAR Dataset', directory='/test/Inertial Signals')


