pytorch 环境命令 conda activate yll
需要下载的包或者库在此环境下下载

整个程序文件明
models---net.py 网络模型结构
NonIID----联邦监督训练文件夹
包含FedAvg FedProx Local MixFedGAN MOON 
outputs ---用于保存模型的文件夹
results---用于保存可视化图像的文件夹
Semi-supervised ---联邦半监督训练文件夹
包含partially_fed DAN ICT MixFedGAN MT VAT
tool ----工具文件夹
包含 Argument.py 设置一些常量
preproceed 原始数据集处理
FedDataset  数据集处理  注意 mixFedGAN下的return与其他方法的返回不一样，需要修改
LungDataset数据集处理  只有一个return


............................................联邦监督训练............................................
main_covid19.py 中fed_main1函数是新冠数据集监督训练的入口，
main_protaste 是前列腺数据集的入口，
covid_ked() 为新冠肺炎数据集直方图的可视化
protaste_ked() 为前列腺图像的密度图显示
plot1() 是可视化数据集的切片，用于展示数据集


当进行本地和集中式训练，请在Arguments.py里设置本地epochs为100
集中式训练：local_train_dataset(args, ensemble_train, ensemble_test)
本地训练：  local_train_dataset(args, train_dataset1, test_dataset1)

当进行联邦学习训练，请在Arguments.py里设置全局轮次round为100，本地epochs为1
联邦监督训练：fed_main1(args, fed_train, fed_test, mode) 
其中mode可以有FedAvg,FedProx,MOON,FedNova,FedBN，MixFedGAN
其中FedProx和MOON正则化项需要调参
fedProx的超参数设置为μ={0.001， 0.01， 0.1， 0.5， 1}，默认一般为0.01
MOON的超参数μ可以设置为1，也可以设置为10.


....................................................联邦半监督训练............................................
main_covid19_semi .py 中fed_main2函数是新冠肺炎半监督训练的入口
crete_client4(train_path1, 256) 
中的train_test_split(lungs, infections, test_size=0.8, random_state=42)
test_size=0.8可以修改数据集的划分

.....................................................模型压缩训练............................................
main_covid19_compression .py 中fed_main3函数是模型压缩训练的入口
channel_pruning_unet ()是对所有层的通道进行剪枝
channel_pruning_unet1() 是本文提出的生成器瓶颈层剪枝
