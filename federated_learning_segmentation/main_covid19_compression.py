# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import warnings

#选择使用第几块GPU
from fed_main3 import fed_main3
from tool.FedDataset import FedDataset
from tool.utils import crete_client2, covid_ked

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings("ignore", category=UserWarning)

from tool.Arguments import Arguments
from pathlib import Path
import torch
from tool.preproceed import preproceed_data
import imgaug.augmenters as iaa

if __name__ == '__main__':

    print(torch.__version__)
    print(torch.version.cuda)
    print("Total GPU Count:{}".format(torch.cuda.device_count()))
    print("Total CPU Count:{}".format(torch.cuda.os.cpu_count()))
    # 获取GPUI设备名称
    print(torch.__version__)
    print(torch.cuda.is_available())

    # 创建一些文件夹方便后续保存输出的图像
    args = Arguments()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    args.result_path1 = os.path.join(args.result_path, args.model_type1)
    if not os.path.exists(args.result_path1):
        os.makedirs(args.result_path1)
    args.result_path2 = os.path.join(args.result_path, args.model_type2)
    if not os.path.exists(args.result_path2):
        os.makedirs(args.result_path2)
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)

    # 此代码是加载新冠肺炎数据集
    # Create the dataset object
    train_path1 = Path("../covid-19_data/Preprocessed/covid19-ct-scans/train/")
    val_path1 = Path("../covid-19_data/Preprocessed/covid19-ct-scans/val/")
    test_path1 = Path("../covid-19_data/Preprocessed/covid19-ct-scans/test/")
    train_dataset1, val_dataset1, test_dataset1 = preproceed_data(train_path1, val_path1, test_path1, flag=True)


    # Create the dataset object
    train_path2 = Path("../covid-19_data/Preprocessed/covid19-40/train/")
    val_path2 = Path("../covid-19_data/Preprocessed/covid19-40/val/")
    test_path2 = Path("../covid-19_data/Preprocessed/covid19-40/test/")
    train_dataset2, val_dataset2, test_dataset2 = preproceed_data(train_path2, val_path2, test_path2, flag=True)


    # Create the dataset object
    train_path3 = Path("../covid-19_data/Preprocessed/covid19-9/train/")
    val_path3 = Path("../covid-19_data/Preprocessed/covid19-9/val/")
    test_path3 = Path("../covid-19_data/Preprocessed/covid19-9/test/")
    train_dataset3, val_dataset3, test_dataset3 = preproceed_data(train_path3, val_path3, test_path3, flag=True)

    # Create the dataset object
    train_path4 = Path("../covid-19_data/Preprocessed/covid19_1110/train/")
    val_path4 = Path("../covid-19_data/Preprocessed/covid19_1110/val/")
    test_path4 = Path("../covid-19_data/Preprocessed/covid19_1110/test/")
    train_dataset4, val_dataset4, test_dataset4 = preproceed_data(train_path4, val_path4, test_path4, flag=True)

    #输出数据集划分数量
    print(len(train_dataset1))
    print(len(test_dataset1))
    print(len(train_dataset2))
    print(len(test_dataset2))
    print(len(train_dataset3))
    print(len(test_dataset3))
    print(len(train_dataset4))
    print(len(test_dataset4))

    fed_train = [train_dataset1, train_dataset2, train_dataset3, train_dataset4]
    fed_test = [test_dataset1, test_dataset2, test_dataset3, test_dataset4]

    # 将四个数据集合并成一个整的数据集用于集合训练
    path = [train_path1,train_path2, train_path3, train_path4]
    #合并所有的dataset
    lungs, infections = crete_client2(path, args.size)
    print(lungs.shape, infections.shape)
    seq = iaa.Sequential([
        iaa.Affine(translate_percent=(0.15),
                   scale=(0.85, 1.15),  # zoom in or out
                   rotate=(-45, 45)  #
                   ),  # rotate up to 45 degrees
        iaa.ElasticTransformation()  # Elastic Transformations
    ])
    ensemble = FedDataset(seq,lungs, infections, args.size, flag=0,weak=True)

    print("-----------------plot----------------")
    #新冠图像的密度图显示
    covid_ked()
    #可视化数据集的切片
    # plot1()

    print("-----------------Compressed model Train NoIID----------------")
    mode='MixFedGAN'
    fed_main3(args, fed_train, fed_test,mode)




