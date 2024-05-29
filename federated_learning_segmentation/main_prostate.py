# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import warnings
import numpy as np
import seaborn as sns

from fed_main1 import fed_main1
from tool.utils import protaste_ked

sns.set(color_codes=True)
np.random.seed(10)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings("ignore", category=UserWarning)

from tool.Arguments import Arguments
from pathlib import Path
import torch
from tool.preproceed import preproceed_data

if __name__ == '__main__':

    print(torch.__version__)
    print(torch.version.cuda)
    print("Total GPU Count:{}".format(torch.cuda.device_count()))
    print("Total CPU Count:{}".format(torch.cuda.os.cpu_count()))
    # 获取GPUI设备名称
    print(torch.__version__)
    print(torch.cuda.is_available())


   #创建文件夹
    args = Arguments()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    args.result_path = os.path.join(args.result_path, args.model_type)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)

    # 此代码是加载前列腺数据集
    # Create the dataset object
    train_path1 = Path("../prostate/Preprocessed/BIDMC/train/")
    val_path1 = Path("../prostate/Preprocessed/BIDMC/test/")
    test_path1 = Path("../prostate/Preprocessed/BIDMC/test/")
    train_dataset1, val_dataset1,test_dataset1 = preproceed_data(train_path1, val_path1,test_path1, flag=True)


    # Create the dataset object
    train_path2 = Path("../prostate/Preprocessed/HK/train/")
    val_path2 = Path("../prostate/Preprocessed/HK/test/")
    test_path2 = Path("../prostate/Preprocessed/HK/test/")
    train_dataset2, val_dataset2 ,test_dataset2= preproceed_data(train_path2, val_path2,test_path2, flag=True)


    # Create the dataset object
    train_path3 = Path("../prostate/Preprocessed/UCL/train/")
    val_path3 = Path("../prostate/Preprocessed/UCL/test/")
    test_path3 = Path("../prostate/Preprocessed/UCL/test/")
    train_dataset3, val_dataset3,test_dataset3 = preproceed_data(train_path3, val_path3,test_path3, flag=True)


    # Create the dataset object
    train_path4 = Path("../prostate/Preprocessed/I2CVB/train/")
    val_path4 = Path("../prostate/Preprocessed/I2CVB/test/")
    test_path4 = Path("../prostate/Preprocessed/I2CVB/test/")
    train_dataset4, val_dataset4,test_dataset4= preproceed_data(train_path4, val_path4,test_path4, flag=True)

    # Create the dataset object
    train_path5 = Path("../prostate/Preprocessed/BMC/train/")
    val_path5 = Path("../prostate/Preprocessed/BMC/test/")
    test_path5 = Path("../prostate/Preprocessed/BMC/test/")
    train_dataset5, val_dataset5,test_dataset5= preproceed_data(train_path5, val_path5,test_path5, flag=True)


    # Create the dataset object
    train_path6 = Path("../prostate/Preprocessed/RUNMC/train/")
    val_path6 = Path("../prostate/Preprocessed/RUNMC/test/")
    test_path6 = Path("../prostate/Preprocessed/RUNMC/test/")
    train_dataset6, val_dataset6,test_dataset6= preproceed_data(train_path6, val_path6,test_path6, flag=True)

    # 数据集划分数量
    print(len(train_dataset1))
    print(len(test_dataset1))
    print(len(train_dataset2))
    print(len(test_dataset2))
    print(len(train_dataset3))
    print(len(test_dataset3))
    print(len(train_dataset4))
    print(len(test_dataset4))
    print(len(train_dataset5))
    print(len(test_dataset5))
    print(len(train_dataset6))
    print(len(test_dataset6))


    fed_train = [train_dataset1, train_dataset2, train_dataset3, train_dataset4,train_dataset5,train_dataset6]
    fed_test = [test_dataset1, test_dataset2, test_dataset3, test_dataset4,test_dataset5,test_dataset6]

    print("-----------------plot----------------")
    #前列腺图像的密度图显示
    protaste_ked()
    # 数据集切片可视化展示
    # plot1()

    print("-----------------Supervised Train NoIID----------------")
    mode = 'FedAvg'
    fed_main1(args, fed_train, fed_test, mode)

