import torch.nn as nn
import mmd

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
            nn.BatchNorm2d(24),
        )
        self.fc1 = nn.Linear(288, 128)
 
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    
class DANNet(nn.Module):

    def __init__(self, num_classes):
        super(DANNet, self).__init__()
        self.cnn = CNN(False)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, source, target):
        loss = 0
        source = self.cnn(source)
        if self.training == True:
            target = self.cnn(target)
            #loss += mmd.mmd_rbf_accelerate(source, target)
            loss += mmd.mmd_rbf_noaccelerate(source, target)

        source = self.fc2(source)
        #target = self.cls_fc(target)

        return source, loss
