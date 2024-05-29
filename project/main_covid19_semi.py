# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import warnings
import numpy as np
import seaborn as sns

from fed_main2 import fed_main2
from tool.utils import crete_client4

sns.set(color_codes=True)
np.random.seed(10)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings("ignore", category=UserWarning)

from tool.Arguments import Arguments
from pathlib import Path
import torch
from tool.FedDataset import FedDataset
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

    # Create directories if not exist
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


    seq = iaa.Sequential([
        iaa.Affine(translate_percent=(0.15),
                   scale=(0.85, 1.15),  # zoom in or out
                   rotate=(-45, 45)  #
                   ),  # rotate up to 45 degrees
        iaa.ElasticTransformation()  # Elastic Transformations
    ])
    model = 'FedAvg'
    data0_client = crete_client4(train_path1, 256)
    client0_data = []
    for i in range(len(data0_client)):
        print(data0_client[i][0].shape, data0_client[i][1].shape)
        data0 = FedDataset(seq,data0_client[i][0], data0_client[i][1], args.size, flag=0,weak=True)
        client0_data.append(data0)

    data1_client = crete_client4(train_path2, 256)
    client1_data = []
    for i in range(len(data1_client)):
        print(data1_client[i][0].shape, data1_client[i][1].shape)
        data1 = FedDataset(seq,data1_client[i][0], data1_client[i][1], args.size, flag=0,weak=True)
        client1_data.append(data1)

    data2_client = crete_client4(train_path3, 256)
    client2_data = []
    for i in range(len(data2_client)):
        print(data2_client[i][0].shape, data2_client[i][1].shape)
        data2 = FedDataset(seq,data2_client[i][0], data2_client[i][1], args.size, flag=0,weak=True)
        client2_data.append(data2)

    data3_client = crete_client4(train_path4, 256)
    client3_data = []
    for i in range(len(data3_client)):
        print(data3_client[i][0].shape, data3_client[i][1].shape)
        data3 = FedDataset(seq,data3_client[i][0], data3_client[i][1], args.size, flag=0,weak=True)
        client3_data.append(data3)

    # plot_dataset(client1_data[0])

    # 分开
    labeled_dataset_train = [client0_data[0], client1_data[0], client2_data[0], client3_data[0]]
    unlabeled_dataset_train = [client0_data[1], client1_data[1], client2_data[1], client3_data[1]]
    fed_test = [test_dataset1, test_dataset2, test_dataset3, test_dataset4]
    fed_train = [train_dataset1, train_dataset2, train_dataset3, train_dataset4]
    print("-----------------Semi-Supervised Train----------------")
    mode = 'mixFedGAN'
    fed_main2(args,labeled_dataset_train,unlabeled_dataset_train,fed_test,mode)
    # mode = 'FedAvg'
    # fed_main1(args, labeled_dataset_train, fed_test, mode)

