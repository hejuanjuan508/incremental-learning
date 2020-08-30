import torch
import random
import scipy.io as scio
import torchvision.transforms as transforms


loader = transforms.Compose([transforms.ToTensor()])
unloader = transforms.ToPILImage()
path = r'C:/Users/55236/PycharmProjects/CNN/data/cwru/12k Drive End Bearing Fault Data/IR028_1.mat'
matdata = scio.loadmat(path)
#print(matdata['X098_DE_time'])
n = 32
for k in range(2000):

    matarray = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        m = random.randint(0, 10000)
        for j in range(n):
            matarray[i][j] = matdata['X057_DE_time'][m+j]
    mattensor = torch.Tensor(matarray)
    mattensor = mattensor.view(32, 32)
    image = transforms.ToPILImage()(mattensor)
    image_path = './data/train/10/' + str(k + 1) + '.jpg'
    image.save(image_path)

