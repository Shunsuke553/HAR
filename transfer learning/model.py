# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 21:16:57 2019

@author: shuns
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
 
    def __init__(self, num_classes):
        super(CNN, self).__init__() # nn.Moduleを継承する
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=18, kernel_size=(2,12)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
            nn.BatchNorm2d(18)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=18, out_channels=36, kernel_size=(1,12)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
            nn.BatchNorm2d(36)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=36, out_channels=24, kernel_size=(1,12)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),
            nn.BatchNorm2d(24)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=6*2*24, out_features=1024), # in_featuresは直前の出力ユニット数
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=num_classes)
        )
 
 
    # Forward計算の定義
    # 参考：Define by Runの特徴（入力に合わせてForward計算を変更可）
    def forward(self, x):
 
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
 
        # 直前のMaxPoolの出力が2次元（×チャネル数）なので，全結合の入力形式に変換
        # 参考：KerasのFlatten()と同じような処理
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        
        return y