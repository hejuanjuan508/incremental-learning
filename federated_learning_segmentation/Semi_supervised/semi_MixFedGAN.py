
import torch.nn.functional as F


from torch import optim as optim

import torch.nn as nn
from torch.autograd import Variable

from tqdm import tqdm
import numpy as np


import torch


from Semi_supervised.util import get_current_consistency_weight, dice_loss, update_ema_variables, cal_gradient_penalty
from tool.utils import CriterionKD, mergeChannels





def test(args,netS,val_loader):

    for i, data in tqdm(enumerate(val_loader, 1)):
        netS.eval()
        with torch.no_grad():
            img, target, gt = Variable(data[0]), Variable(data[1]), Variable(data[2])
            img = img.type(torch.FloatTensor)
            gt = gt.type(torch.FloatTensor)
            img= img.cuda()
            #target = target.cuda()
            gt = gt.cuda()
            pred= netS(img)
            pred = torch.sigmoid(pred)

            pred_np = torch.from_numpy(mergeChannels(pred.data.cpu().numpy(), img.shape[2])).cuda()

            # print(pred_np.shape)

            N = gt.size(0)
            pred_flat = pred_np.view(N, -1)
            gt_flat = gt.view(N, -1)
            eps = 0.0000001

            tn = torch.sum((1 - gt_flat) * (1 - pred_flat), dim=1)
            tp = torch.sum(gt_flat * pred_flat, dim=1)
            fp = torch.sum(pred_flat, dim=1) - tp
            fn = torch.sum(gt_flat, dim=1) - tp
            loss_di = (2 * tp + eps) / (2 * tp + fp + fn + eps)
            mdice = loss_di.sum() / N
    return mdice

def train_semi_MixFedGAN(args, lr,client, netS_local, netC_local,meme, labeled_train_loader,val_loader,fed_round, unlabed_train_loader):
    print("This is Mean Teacher")
    print("客户端{}开始训练".format(client))
    unlabel_iter = iter(unlabed_train_loader)
    len_iter = len(unlabel_iter)
    label_iter = iter(labeled_train_loader)
    # print(len_iter) 216
    Dice = []
    criterion_kd = CriterionKD().cuda()
    if args.use_cuda:
        netS_local = netS_local.cuda()
        netC_local = netC_local.cuda()
        meme = meme.cuda()
    print('learning-rate{}'.format(lr))
    iter_num = 0
    netS_local.train()  # 学生模型初始化
    meme.train()  # 教师模型初始化
    optimizerG = optim.Adam(netS_local.parameters(), lr=lr, betas=(args.beta1, args.beta2))
    optimizerD = optim.Adam(netC_local.parameters(), lr=lr, betas=(args.beta1, args.beta2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(1, args.epochs + 1):
        for batch_idx in range(len_iter):
            # 梯度请0
            netS_local.zero_grad()
            # 获取带标签和无标签的数据
            try:
                image_s1, target_s1, gt_s1, image_s2, target_s2, gt_s2 = next(label_iter)

            except StopIteration:
                labeled_iter = iter(labeled_train_loader)
                image_s1, target_s1, gt_s1, image_s2, target_s2, gt_s2 = next(labeled_iter)

            try:
                image_u, _, _, _, _, _ = next(unlabel_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabed_train_loader)
                image_u, _, _, _, _, _ = next(unlabeled_iter)

            image_s1 = image_s1.float()
            target_s1 = target_s1.float()
            image_u = image_u.float()
            image_s2 = image_s2.float()
            if args.use_cuda:
                image_s1 = image_s1.cuda()
                target_s1 = target_s1.cuda()
                image_u = image_u.cuda()
                image_s2 = image_s2.cuda()

            output_s1 = netS_local(image_s1)
            output_s1 = F.sigmoid(output_s1)
            output_s1 = output_s1.detach()



            # 计算多尺度损失函数
            # output_masked = image.clone()
            input_mask_s1 = image_s1.clone()


            output_masked_s1 = input_mask_s1 * output_s1
            if args.use_cuda:
                output_masked_s1 = output_masked_s1.cuda()
            # target_masked = image.clone()
            target_masked_s1 = input_mask_s1 * target_s1
            if args.use_cuda:
                target_masked_s1 = target_masked_s1.cuda()

            output_D1 = netC_local(output_masked_s1)
            target_D1 = netC_local(target_masked_s1)
            loss_D = 1 - torch.mean(torch.abs(output_D1 - target_D1))

            loss_D_gp = cal_gradient_penalty(netC_local, target_s1, output_s1, device, type='mixed', constant=1.0, lambda_gp=10.0)[0]
            loss_D_joint = loss_D + args.lambda_d * loss_D_gp

            loss_D_joint.backward()
            optimizerD.step()

            netS_local.zero_grad()

            output_s1 = netS_local(image_s1)
            output_s1 = F.sigmoid(output_s1)

            with torch.no_grad():

                output_s2 = netS_local(image_s2)
                output_s2 = F.sigmoid(output_s2)

                output_s3 = netS_local(image_u)
                output_s3 = F.sigmoid(output_s3)

                output_s4 = meme(image_u)
                output_s4=F.sigmoid(output_s4)

            kl_local = criterion_kd(output_s3, output_s4)
            lcon = criterion_kd(output_s1, output_s2)
            # 计算多尺度L1损失函数
            # output_masked = image.clone()
            input_mask_s1 = image_s1.clone()

            output_masked_s1 = input_mask_s1 * output_s1

            if args.use_cuda:
                output_masked_s1 = output_masked_s1.cuda()

            # target_masked = image.clone()
            target_masked_s1 = input_mask_s1 * target_s1

            if args.use_cuda:
                target_masked_s1 = target_masked_s1.cuda()

            output_G1 = netC_local(output_masked_s1)
            target_G1 = netC_local(target_masked_s1)

            loss_G1 = torch.mean(torch.abs(output_G1 - target_G1))
            loss_dice = dice_loss(output_s1, target_s1)

            loss_G_joint = loss_G1 + args.alpha * loss_dice+0.1*kl_local+0.1*lcon

            loss_G_joint.backward()
            optimizerG.step()

            if (batch_idx % 10 == 0):
                print("\nEpoch[{}/{}]\tBatch({}/{}):\tG_Loss: {:.4f}\tD_Loss: {:.4f} \n".format(
                    fed_round,  args.rounds, batch_idx,  len_iter, loss_G1.item(), loss_D.item()))
            # saving visualizations after each epoch to monitor model's progress
        dice = test(args, netS_local, val_loader)
        params1 = list(meme.parameters())
        params2 = list(netS_local.parameters())
        distance = 0.0
        for w, w_t in zip(params1, params2):
            distance += torch.norm(w - w_t)
        distance /= len(params1)  # 求平均距离
        # 归一化确保相似性在[0,1]之间
        similarity = 1.0 / (1.0 + distance)  # 相似性度量
    return netS_local,netC_local,dice,similarity