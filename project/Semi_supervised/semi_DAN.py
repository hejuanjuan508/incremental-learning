from torch import optim as optim, Tensor
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss

from Semi_supervised.util import dice_loss, get_current_consistency_weight, softmax_mse_loss, update_ema_variables, \
    cal_gradient_penalty


def train_semi_DAN(args,lr,client,netS_local,netC_local,labeled_train_loader,fed_round,unlabed_train_loader):
    print("This is DAN")
    print("客户端{}开始训练".format(client))
    unlabel_iter = iter(unlabed_train_loader)
    len_iter = len(unlabel_iter)
    label_iter = iter(labeled_train_loader)
    # print(len_iter) 216
    Dice = []

    if args.use_cuda:
        netS = netS_local.cuda()
        netC = netC_local.cuda()
    print('learning-rate{}'.format(lr))
    iter_num = 0
    loss_function = nn.BCELoss()
    if fed_round>50:
        args.lambda_adv=1

    #netS.train()
    optimizerG = optim.Adam(netS.parameters(), lr=lr, betas=(args.beta1, args.beta2))
    optimizerD = optim.Adam(netC.parameters(), lr=lr, betas=(args.beta1, args.beta2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(1, args.epochs + 1):
        for batch_idx in range(len_iter):
            # 梯度请0
            netC.zero_grad()
            # 获取带标签和无标签的数据
            try:
                image_s1, target_s1,gt_s1 = next(label_iter)

            except StopIteration:
                labeled_iter = iter(labeled_train_loader)
                image_s1,target_s1, gt_s1= next(labeled_iter)

            try:
                image_u, _, _ = next(unlabel_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabed_train_loader)
                image_u, _, _= next(unlabeled_iter)

            valid = Variable(Tensor(image_s1.size(0), 2).fill_(1), requires_grad=False).cuda()
            fake = Variable(Tensor(image_u.size(0), 2).fill_(0), requires_grad=False).cuda()

            image_s1 = image_s1.float()
            target_s1 = target_s1.float()
            image_u = image_u.float()

            if args.use_cuda:
                image_s1 = image_s1.cuda()
                target_s1 = target_s1.cuda()
                image_u = image_u.cuda()

            #有注释
            output_s1 = netS(image_s1)
            output_s1 = F.sigmoid(output_s1)
            output_s1 = output_s1.detach()

            #未注释
            output_s2 = netS(image_u)
            output_s2 = F.sigmoid(output_s2)
            output_s2 = output_s2.detach()

            # 计算多尺度损失函数
            # output_masked = image.clone()
            input_mask_s1 = image_s1.clone()
            input_mask_u = image_u.clone()

            output_masked_s1 = input_mask_s1 * output_s1
            output_masked_s2 = input_mask_u * output_s2
            if args.use_cuda:
                output_masked_s1 = output_masked_s1.cuda()
                output_masked_s2 = output_masked_s2.cuda()
            # target_masked = image.clone()
            target_masked_s1 = input_mask_s1 * target_s1
            if args.use_cuda:
                target_masked_s1 = target_masked_s1.cuda()

            #将注释的图像视为真
            #print(output_masked_s1.shape, output_s1.shape, image_s1.shape)
            output_D1,pred_real = netC(output_masked_s1,output_s1,image_s1)
            #print(pred_real.shape, valid.shape)
            loss_real = loss_function(pred_real, valid)
            # 将未注释的图像视为假
            #print(output_masked_s2.shape, output_s2.shape, image_u.shape)
            output_D2, pred_fake= netC(output_masked_s2, output_s2, image_u)
            #print(pred_fake.shape, fake.shape)
            loss_fake = loss_function(pred_fake, fake)

            target_D1,_= netC(target_masked_s1,target_s1,image_s1)

            loss_D = 1 - torch.mean(torch.abs(output_D1 - target_D1))

            loss_D_gp = cal_gradient_penalty(netC,image_s1, target_s1, output_s1, device, type='mixed', constant=1.0, lambda_gp=10.0)[
                0]
            loss_D_joint = loss_D + args.lambda_d * loss_D_gp +args.lambda_adv*(loss_fake+loss_real)

            loss_D_joint.backward()

            # loss_D.backward()

            optimizerD.step()

            # 训练分割网

            netS.zero_grad()

            output_s1 = netS(image_s1)
            output_s1 = F.sigmoid(output_s1)

            output_s2=netS(image_u)
            output_s2=F.sigmoid(output_s2)


            # 计算多尺度L1损失函数
            # output_masked = image.clone()
            output_masked_s1 = input_mask_s1 * output_s1
            output_masked_s2 = input_mask_u * output_s2
            if args.use_cuda:
                output_masked_s1 = output_masked_s1.cuda()
                output_masked_s2 = output_masked_s2.cuda()
            # target_masked = image.clone()
            target_masked_s1 = input_mask_s1 * target_s1
            if args.use_cuda:
                target_masked_s1 = target_masked_s1.cuda()
            valid = Variable(Tensor(image_u.size(0), 2).fill_(1), requires_grad=False).cuda()
            #在这里文献里面提到的是将未注释的标签视为真
            output_G, pred_fake = netC(output_masked_s2, output_s2, image_u)
            loss_adv_G = loss_function(pred_fake, valid)
            output_G1,_ = netC(output_masked_s1,output_s1,image_s1)
            target_G1,_ = netC(target_masked_s1,target_s1,image_s1)
            loss_G1 = torch.mean(torch.abs(output_G1 - target_G1))
            loss_dice = dice_loss(output_s1, target_s1)

            loss_G_joint = loss_G1 + args.alpha * loss_dice+ args.lambda_adv*loss_adv_G

            loss_G_joint.backward()
            optimizerG.step()

            if (batch_idx % 10 == 0):
                print("\nEpoch[{}/{}]\tBatch({}/{}):\tG_Loss: {:.4f}\tD_Loss: {:.4f} \n".format(
                    fed_round,  args.rounds, batch_idx,  len_iter, loss_G_joint.item(), loss_D.item()))
            # saving visualizations after each epoch to monitor model's progress

    return netS,netC