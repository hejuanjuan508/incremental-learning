import os
import time
from torchsummary import summary
from NonIID.FedAvg import FedAvg_ClientUpdate
from NonIID.FedProx import FedProx_ClientUpdate
from NonIID.MOON import MOON_ClientUpdate
from NonIID.MixFedGAN import MixFedGAN_ClientUpdate
from fed_test import fed_test
from models.net import NetS, NetC
import copy
import torch.backends.cudnn as cudnn
import torch


def create_model(ndf,ema=False):
    model =NetS(ndf)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def nova_aggregation(client_size):
    #sampled_datasize=sum(list(client_size.values()))
    t_k=[int(i/4) for i in client_size]  #[343, 20, 70, 147]
    print(t_k)
    # dg_weight=[i*j/sampled_datasize for i, j in zip(client_size, t_k)]
    # local_weight=[i/(sampled_datasize*j) for i, j in zip(client_size, t_k)]
    novaWeight=[i * j for i, j in zip(client_size, t_k)]
    print(novaWeight)
    novaWeight = [ round((sum(client_size)*sum(t_k)/i),2) for i in novaWeight]

    normalized_novaWeight = [round(i/sum(novaWeight),2) for i in novaWeight]


    # 计算最小值和最大值
    # min_weight = min(novaWeight)
    # max_weight = max(novaWeight)
    # # 归一化到 [0, 1] 范围内
    # novaWeight = [round((i - min_weight) / (max_weight - min_weight),2) for i in novaWeight]

    print(normalized_novaWeight)
    return normalized_novaWeight

def get_weight(mode,Scorce1,Scorce2,dataset_train):
    client=4
    client_size=[]
    if mode=='FedAvg':
        weight=[1. / client for i in range(client)]
    if mode=='FedProx':
        weight=[1. / client for i in range(client)]
    if mode=='MOON':
        weight=[1. / client for i in range(client)]
    if mode=='FedBN':
        weight=[1. / client for i in range(client)]
    if mode=='FedNova':
        for idx in range(len(dataset_train)):
            client_size.append(len(dataset_train[idx]))

        weight = nova_aggregation(client_size)

    if mode == 'MixFedGAN':
        with torch.no_grad():
            temp1=sum(Scorce1)
            temp2=sum(Scorce2)
            temp3 = 0
            a = 0.8
            b = 0.2
            print("聚合")
            # 计算模型的偏移
            weight = [Scorce1[0] / temp1, Scorce1[1] / temp1, Scorce1[2] / temp1, Scorce1[3] / temp1]


            list2 = [Scorce2[0] / temp2, Scorce2[1] / temp2, Scorce2[2] / temp2, Scorce2[3] / temp2]
            print(list2)

            for i in range(4):
                temp3 += (weight[i].cpu() * a + list2[i].cpu() * b)
                # 缩放
            for i in range(4):
                weight[i] = (weight[i].cpu() * a + list2[i].cpu() * b) / temp3

    return weight

def fed_main1(args,dataset_train,dataset_test,mode):
    # 新冠数据集客户端
    supervised_user_id = [0,1,2,3]
    # 前列腺数客户端
    # supervised_user_id = [0, 1, 2, 3, 4, 5]
    ndf=32
    #设置全局模型：
    netS_server=create_model(ndf)
    netC_server= NetC(ndf)


    lr=args.lr

    Dice=[[],[],[],[]]
    history_w = []
    history_w.append(netS_server)
    Old_List_S = [copy.deepcopy(netS_server) for idx in range(len(supervised_user_id))]


    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    start_time = time.time()
    for fed_round in range(1,args.rounds+1):

        w_locals=[]
        c_locals = []
        Scorce1 = []
        Scorce2 = []


        print("-------------全局第{}轮-----------------".format(fed_round))


        for idx in supervised_user_id:

            train_loader = torch.utils.data.DataLoader(dataset_train[idx],
                                                 batch_size=args.train_batch_size,
                                                 shuffle=True)
            val_loader = torch.utils.data.DataLoader(dataset_test[idx],
                                                       batch_size=args.train_batch_size,
                                                       shuffle=False)
            # 学习率衰减
            if fed_round % 25 == 0:
                lr = lr * args.decay_rate
                if lr <= 0.00000001:
                    lr = 0.00000001

                print('learning-rate{}'.format(lr))

            #step1:本地客户端接受从服务器发送过来的模型

            netS_local = copy.deepcopy(netS_server)
            netC_local = copy.deepcopy(netC_server)

            # step2:客户端本地训练
            if mode=='FedAvg':
                netS_local, netC_local=FedAvg_ClientUpdate(args, lr, idx, netS_local,netC_local, train_loader, fed_round)
                w_locals.append(netS_local)
                c_locals.append(netC_local)

            if mode=='FedProx':
                netS_local, netC_local=FedProx_ClientUpdate(args, lr, idx, netS_local,netC_local,history_w[0], train_loader, fed_round)
                w_locals.append(netS_local)
                c_locals.append(netC_local)

            if mode=='MOON':
                #MOON需要上一轮的全局模型和每个客户端上一轮的全局模型
                netS_local, netC_local=MOON_ClientUpdate(args, lr, idx, netS_local,netC_local,history_w[0], Old_List_S[idx], train_loader, fed_round)
                w_locals.append(netS_local)
                c_locals.append(netC_local)
                # pre_S[idx].load_state_dict(copy.deepcopy(netS_local))  # 加载的模型不要用变量去接
                Old_List_S[idx] = netS_local  # 每一个客户端的历史模型等于上一局的局部模型

            if mode=='FedBN':
                netS_local, netC_local=FedAvg_ClientUpdate(args, lr, idx, netS_local,netC_local, train_loader, fed_round)
                w_locals.append(netS_local)
                c_locals.append(netC_local)

            if mode=='FedNova':
                netS_local, netC_local=FedAvg_ClientUpdate(args, lr, idx, netS_local,netC_local, train_loader, fed_round)
                w_locals.append(netS_local)
                c_locals.append(netC_local)
                # 每个客户端数据集的大小


            if mode == 'MixFedGAN':
                netS_local, netC_local,scorce1,scorce2=MixFedGAN_ClientUpdate(args, lr, idx,netS_local,netC_local,history_w[0] ,train_loader,val_loader,fed_round)
                w_locals.append(netS_local)
                c_locals.append(netC_local)
                #保存模型偏移量
                Scorce1.append(scorce1.cpu())
                #保存模型本地测试精度值
                Scorce2.append(scorce2)


        # step3: 服务器端聚合
        w_glob = netS_local.state_dict()
        c_glob = netC_local.state_dict()

        weight=get_weight(mode,Scorce1,Scorce2,dataset_train)
        print(weight)
        #如果是FedBN执行下面代码
        if mode.lower()=='fedbn':
            print('this is FedBN')
            for k in w_glob.keys():
                #过滤掉BN层的模型参数
                if 'norm' not in k:
                    w_glob[k] = torch.zeros_like(netS_local.state_dict()[k].cpu()).float()
                    for i in range(0, len(w_locals)):
                        w_glob[k] += torch.mul(w_locals[i].state_dict()[k].cpu(), weight[i])
            for k in c_glob.keys():
                if 'norm' not in k:
                    c_glob[k] = torch.zeros_like(netC_local.state_dict()[k]).float()
                    for i in range(0, len(c_locals)):
                        c_glob[k] += torch.mul(c_locals[i].state_dict()[k].cpu(), weight[i])

        #如果不是FedBN执行下面代码
        else:
            print("this is ",mode)
            for k in w_glob.keys():
                w_glob[k] = torch.zeros_like(netS_server.cpu().state_dict()[k].cpu()).float()
                for i in range(0, len(w_locals)):
                    w_glob[k] += torch.mul(w_locals[i].state_dict()[k].cpu(), weight[i])
            for k in c_glob.keys():
                c_glob[k] = torch.zeros_like(netC_server.cpu().state_dict()[k]).float()
                for i in range(0, len(w_locals)):
                    c_glob[k] += torch.mul(c_locals[i].state_dict()[k].cpu(), weight[i])

        # step4: 模型分发
        netS_server.load_state_dict(w_glob)
        netC_server.load_state_dict(c_glob)
        #获得上一轮全局模型
        history_w.append(netS_server)
        del history_w[0]

        for idx in range(len(supervised_user_id)):
            test_loader = torch.utils.data.DataLoader(dataset_test[idx],batch_size = args.val_batch_size,shuffle = False)
            dice=fed_test(args, netS_server, test_loader, fed_round,idx,flag=1)
            Dice[idx].append(dice)
            print(Dice[idx])
        torch.cuda.empty_cache()
        print("epoch Stopped:{:.2f}".format(time.time() - start_time))

        if (fed_round % 100 == 0):
            torch.save(netS_server.state_dict(), 'net.pth')
            summary(netS_server.cuda(), input_size=(1, 256, 256), batch_size=4)
            summary(netC_server.cuda(), input_size=(2, 256, 256), batch_size=4)
