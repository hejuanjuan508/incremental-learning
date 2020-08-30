import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

batch_size = 100
learning_rate = 0.008
num_epochs = 10

def loadtraindata():
    path = "./data/train"  #路径
    trainset = torchvision.datasets.ImageFolder(
        path,
        transform=transforms.Compose([#transforms.Compose()是把几个transform语句合并到一起
            transforms.Resize((32, 32)),#将把图片缩放到（h,w）大小（长宽比会改变）。另一类是传入单个参数，这将把图片经过缩放后（保持长宽比不变），将最短的边缩放到传入的参数。
            transforms.CenterCrop(32),
            transforms.ToTensor(),#数据类型的转换 归一化到(0,1)
        ]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    return trainloader


def loadtestdata():
    path = "./data/test"
    testset = torchvision.datasets.ImageFolder(
        path, transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)
    return testloader



class CNN(nn.Module):  #定义网络
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2), #nn.Conv2d对由多个输入平面组成的输入信号进行二维卷积
            nn.BatchNorm2d(16),
            nn.ReLU(),                                 #激活层
            nn.MaxPool2d(2))                           #池化层
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(2 * 2 * 128, 10)  #全连接层

    def forward(self, x):                     #前向传播
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)#.view( )是一个tensor的方法，使得tensor改变size但是元素的总数是不变的。
        out = self.fc(x)
        return out


def trainandsave():    #训练
    trainloader = loadtraindata()
    cnn = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            # Forward and backward
            outputs = cnn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print('Epoch:[%d/%d],Step:[%d/%d],Loss:%.4f' % (
                    epoch + 1, num_epochs, i + 1, 20000 // batch_size, loss.item()))

    print('Finished Training')
    torch.save(cnn, 'cnn.pkl')
    torch.save(cnn.state_dict(), 'cnn_params.pkl')


def reload_net():
    trainednet = torch.load('cnn.pkl')
    return trainednet


def test():  #测试
    testloader = loadtestdata()
    cnn = reload_net()
    cnn.eval()
    correct = 0
    total = 0
    for j, (images, labels) in enumerate(testloader):
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.shape[0]
        correct += (predicted == labels).sum()
    print('训练模型模型的精度: %d %%' % (100 * correct / total))


def main():
    #trainandsave()
    test()


if __name__ == '__main__':
    main()

