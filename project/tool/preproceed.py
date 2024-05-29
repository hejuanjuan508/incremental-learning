from pathlib import Path


# from sklearn.model_selection import train_test_split
# import syft as sy
import imgaug.augmenters as iaa

from tool.LungDataset import LungDataset
from tool.utils import crete_split, crete_split1


def preproceed_data(train_path, val_path,test_path,flag):
    seq = iaa.Sequential([
        iaa.Affine(translate_percent=(0.15),
                   scale=(0.85, 1.15),  # zoom in or out
                   rotate=(-45, 45)  #
                   ),  # rotate up to 45 degrees
        iaa.ElasticTransformation()  # Elastic Transformations
    ])
    # 获取四个数据集所在文件夹
    root1 = Path("../covid-19_data/covid19-ct-scans/ct_scan/")
    label1 = Path("../covid-19_data/covid19-ct-scans/infection_mask/")
    root2 = Path("../covid-19_data/covid19-40/ct_scan/")
    val = Path("../covid-19_data/covid19-40/val/")
    root3 = Path("../covid-19_data/covid19-9/ct_scan/")
    label3 = Path("../covid-19_data/covid19-9/infection_mask/")
    root4 = Path("../covid-19_data/covid19_1110/ct_scan/")
    label4 = Path("../covid-19_data/covid19_1110/infection_mask/")

#非监督的客户端
    root_client0 = Path("../covid-19_data/covid19_1110/studies/CT-0/")
    root_client1 = Path("../covid-19_data/covid19_1110/studies/CT-1/")
    root_client2 = Path("../covid-19_data/covid19_1110/studies/CT-2/")
    root_client3 = Path("../covid-19_data/covid19_1110/studies/CT-3/")
    root_client4 = Path("../covid-19_data/covid19_1110/studies/CT-4/")
    root_client5 = Path("../covid-19_data/covid19_1110/studies/CT-5/")
    root_client6 = Path("../covid-19_data/covid19_1110/studies/CT-6/")
    root_client7 = Path("../covid-19_data/covid19_1110/studies/CT-7/")
    root_client8 = Path("../covid-19_data/covid19_1110/studies/CT-8/")
    root_client9 = Path("../covid-19_data/covid19_1110/studies/CT-9/")

    # 获取文件夹下所有.nii文件
    all_files1 = list(root1.glob("lung*"))  # Get all subjects
    all_files2 = list(root2.glob("tr*"))  # Get all subjects
    val_files = list(val.glob("val*"))
    all_files3 = list(root3.glob("lung*"))  # Get all subjects
    all_files4 = list(root4.glob("study*"))  # Get all subjects

    all_client0=list(root_client0.glob("study*"))
    all_client1 = list(root_client1.glob("study*"))
    all_client2 = list(root_client2.glob("study*"))
    all_client3 = list(root_client3.glob("study*"))
    all_client4 = list(root_client4.glob("study*"))
    all_client5 = list(root_client5.glob("study*"))
    all_client6 = list(root_client6.glob("study*"))
    all_client7 = list(root_client7.glob("study*"))
    all_client8 = list(root_client8.glob("study*"))
    all_client9 = list(root_client9.glob("study*"))




    # 数据集1：该数据集是20例诊断为covid-19的患者的CT扫描以及专家对肺部和感染的分割
    save_root1 = Path("../../covid-19_data/Preprocessed/covid19-ct-scans")
    crete_split(all_files1, all_files2, all_files4, save_root1)

    # 数据集2：该数据集是40名COVID-19患者的100张轴向CT图像的数据集，这些图像是从此处找到的可公开访问的JPG图像转换而来的
    save_root2 = Path("../../covid-19_data/Preprocessed/covid19-40")
    crete_split(all_files2, all_files2, all_files4, save_root2)
    # 数据集3:包括整个体积，因此包括正片和负片（放射科医生已将829个切片中的373个评估为阳性和分段）
    save_root3 = Path("../../covid-19_data/Preprocessed/covid19-9")
    crete_split(all_files3, all_files2, all_files4, save_root3)
    # 数据集4：有50个病人的数据集，每个病人的切片并不多
    save_root4 = Path("../../covid-19_data/Preprocessed/covid19_1110")
    crete_split(all_files4, all_files2, all_files4, save_root4)

    # 客户端0~9用做无监督训练
    save_client_root0 = Path("../covid-19_data/Preprocessed/client0")
    crete_split1(all_client0, save_client_root0)
    save_client_root1 = Path("../covid-19_data/Preprocessed/client1")
    crete_split1(all_client1,  save_client_root1)
    save_client_root2 = Path("../covid-19_data/Preprocessed/client2")
    crete_split1(all_client2, save_client_root2)
    save_client_root3 = Path("../covid-19_data/Preprocessed/client3")
    crete_split1(all_client3, save_client_root3)
    save_client_root4 = Path("../covid-19_data/Preprocessed/client4")
    crete_split1(all_client4,  save_client_root4)
    save_client_root5 = Path("../covid-19_data/Preprocessed/client5")
    crete_split1(all_client5, save_client_root5)
    save_client_root6 = Path("../covid-19_data/Preprocessed/client6")
    crete_split1(all_client6,save_client_root6)
    save_client_root7 = Path("../covid-19_data/Preprocessed/client7")
    crete_split1(all_client7,  save_client_root7)
    save_client_root8 = Path("../covid-19_data/Preprocessed/client8")
    crete_split1(all_client8, save_client_root8)
    save_client_root9 = Path("../covid-19_data/Preprocessed/client9")
    crete_split1(all_client9, save_client_root9)

    model= 'mixFedGAN'
#说明flag==True表示数据都是有标签的 ，flag1==True表示数据进行若增强和强增强，weak和strong是开启弱增强和强增强
    train_dataset = LungDataset(seq,model,train_path, flag, size=256,flag1=True,weak=True,strong=True,transform=True)
    val_dataset = LungDataset(seq,model,val_path, flag,size=256,flag1=True,weak=True,strong=True, transform=True)
    test_dataset = LungDataset(seq, model, test_path, flag, size=256, flag1=True, weak=True, strong=True, transform=True)


    return train_dataset, val_dataset,test_dataset


