import os
import time
from torchsummary import summary

from Semi_supervised.partially_fed import ClientUpdate
from Semi_supervised.semi_DAN import train_semi_DAN
from Semi_supervised.semi_ICT import train_semi_ICT
from Semi_supervised.semi_MT import train_semi_MT
from Semi_supervised.semi_MixFedGAN import train_semi_MixFedGAN
from Semi_supervised.semi_VAT import train_semi_VAT
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



def get_weight(mode,Scorce1,Scorce2):
    client=4
    if mode!='mixFedGAN':
        weight=[1. / client for i in range(client)]
    if mode == 'mixFedGAN':
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

def fed_main2(args,labeled_data_train,unlabeled_dataset_train,dataset_test,mode):
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

            labeled_data_loader = torch.utils.data.DataLoader(labeled_data_train[idx],
                                                 batch_size=args.train_batch_size,
                                                 shuffle=True)
            unlabed_data_loader = torch.utils.data.DataLoader(unlabeled_dataset_train[idx],
                                                       batch_size=args.train_batch_size,
                                                       shuffle=False)
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
            ema_model = create_model(ndf)
            netC_local = copy.deepcopy(netC_server)

            # step2:客户端本地训练
            if mode=='only':
                netS_local, netC_local=ClientUpdate(args, lr, idx, netS_local,netC_local, labeled_data_loader, fed_round)
                w_locals.append(netS_local)
                c_locals.append(netC_local)

            if mode == 'mt':
                netS_local, netC_local = train_semi_MT(args, lr, idx, netS_local, netC_local, ema_model,
                                                       labeled_data_loader,
                                                       fed_round, unlabed_data_loader)
                w_locals.append(netS_local)
                c_locals.append(netC_local)

            if mode == 'vat':
                netS_local, netC_local = train_semi_VAT(args, lr, idx, netS_local, netC_local, ema_model,
                                                        labeled_data_loader,
                                                        fed_round, unlabed_data_loader)
                w_locals.append(netS_local)
                c_locals.append(netC_local)
            if mode == 'dan':
                netS_local, netC_local = train_semi_DAN(args, lr, idx, netS_local, netC_local, labeled_data_loader,
                                                        fed_round, unlabed_data_loader)
                w_locals.append(netS_local)
                c_locals.append(netC_local)

            if mode == 'ict':
                netS_local, netC_local = train_semi_ICT(args, lr, idx, netS_local, netC_local, ema_model,
                                                        labeled_data_loader,
                                                        fed_round, unlabed_data_loader)
                w_locals.append(netS_local)
                c_locals.append(netC_local)

            if mode == 'mixFedGAN':
                netS_local, netC_local, scorce1, scorce2 = train_semi_MixFedGAN(args, lr, idx, netS_local, netC_local,
                                                                                history_w[0], labeled_data_loader,val_loader,
                                                                                fed_round, unlabed_data_loader)

                w_locals.append(netS_local)
                c_locals.append(netC_local)
                # 保存模型偏移量
                Scorce1.append(scorce1.cpu())
                # 保存模型本地测试精度值
                Scorce2.append(scorce2)

        # step3: 服务器端聚合
        w_glob = netS_local.state_dict()
        c_glob = netC_local.state_dict()

        weight=get_weight(mode,Scorce1,Scorce2)
        print(weight)
        #如果是FedBN执行下面代码
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
