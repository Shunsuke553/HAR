import torch
import math

# 学習用関数
def train(source_loader, target_train_loader, model, optimizer, loss_fn, device, total_epoch, epoch):
    model.train()
    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)
    num_iter = len(source_loader)
    for i in range(1, num_iter):
        data_source, label_source = iter_source.next()
        data_target, _ = iter_target.next()
        if i % len(target_train_loader) == 0:
            iter_target = iter(target_train_loader)
 
        data_source = data_source.to(device)
        label_source = label_source.to(device)
        data_target = data_target.to(device)
 
        optimizer.zero_grad()
        label_source_pred, loss_mmd = model(data_source, data_target)
        loss_cls = loss_fn(label_source_pred, label_source)
        gamma = 2 / (1 + math.exp(-10 * (epoch) / total_epoch)) - 1
        loss = loss_cls + gamma*loss_mmd
        loss.backward()
        optimizer.step()
    
    print ('Epoch [%d/%d], Loss: %.4f' % (epoch, total_epoch, loss.item()))
    
# テスト用関数
def test(target_test_loader, model, device):
    model.eval() # モデルを推論モードに変更
    correct = 0
    with torch.no_grad(): # 推論時には勾配は不要
        for data, targets in target_test_loader:
            data = data.to(device)
            targets = targets.to(device)
            output, _ = model(data, data)
            _, predicted = torch.max(output.data, 1) # 確率が最大のラベルを取得
            correct += predicted.eq(targets.data.view_as(predicted)).sum() # 正解ならば正解数をカウントアップ
    # 正解率を計算
    data_num = len(target_test_loader.dataset) # テストデータの総数
    print('\nAccuracy: {}/{} ({:.0f}%)\n'.format(correct, data_num, 100. * correct / data_num))
