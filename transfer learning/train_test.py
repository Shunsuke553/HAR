# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 21:20:46 2019

@author: shuns
"""
import torch

# 学習用関数
def train(loader_train, model_obj, optimizer, loss_fn, device, total_epoch, epoch):
    
    model_obj.train() # モデルを学習モードに変更
 
    # ミニバッチごとに学習
    for data, targets in loader_train:
 
        data = data.to(device) # GPUを使用するため，to()で明示的に指定
        targets = targets.to(device) # 同上
 
        optimizer.zero_grad() # 勾配を初期化
        outputs = model_obj(data) # 順伝播の計算
        loss = loss_fn(outputs, targets) # 誤差を計算
        loss.backward() # 誤差を逆伝播させる
        optimizer.step() # 重みを更新する
    
    print ('Epoch [%d/%d], Loss: %.4f' % (epoch, total_epoch, loss.item()))
    
# テスト用関数
def test(loader_test, trained_model, device):
 
    trained_model.eval() # モデルを推論モードに変更
    correct = 0 # 正解率計算用の変数を宣言
 
    # ミニバッチごとに推論
    with torch.no_grad(): # 推論時には勾配は不要
        for data, targets in loader_test:
 
            data = data.to(device) #  GPUを使用するため，to()で明示的に指定
            targets = targets.to(device) # 同上
 
            outputs = trained_model(data) # 順伝播の計算
 
            # 推論結果の取得と正誤判定
            _, predicted = torch.max(outputs.data, 1) # 確率が最大のラベルを取得
            correct += predicted.eq(targets.data.view_as(predicted)).sum() # 正解ならば正解数をカウントアップ
    
    # 正解率を計算
    data_num = len(loader_test.dataset) # テストデータの総数
    print('\nAccuracy: {}/{} ({:.0f}%)\n'.format(correct, data_num, 100. * correct / data_num))
