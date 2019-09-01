import warnings
warnings.filterwarnings('ignore')
import torch
from torch import optim,nn

from models import DANNet
from train_test import train, test
from data_loader import load_training, load_testing


# GPUの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# ハイパーパラメータの設定
batch_size = 256
num_classes = 6
epochs = 10

# データのロード
source_loader = load_training(root_path='../UCI HAR Dataset', directory='/train/Inertial Signals', batch_size=256)
target_train_loader = load_training(root_path='../UCI HAR Dataset', directory='/test/Inertial Signals', batch_size=256)
target_test_loader = load_testing(root_path='../UCI HAR Dataset', directory='/test/Inertial Signals')

# モデル作成
model = DANNet(num_classes=num_classes).to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
 
# train. test
print('Begin train')
for epoch in range(1, epochs+1):
    train(source_loader, target_train_loader, model, optimizer, loss_fn, device, epochs, epoch)
    test(target_test_loader, model, device)