from torch import optim as optim
import torch
import torch.nn.functional as F

from Semi_supervised.util import vat_loss, dice_loss, cal_gradient_penalty


def train_semi_VAT(args,lr,client,netS_local,netC_local,ema_model,labeled_train_loader,fed_round,unlabed_train_loader):
    print("This is VAT")
    print("客户端{}开始训练".format(client))
    unlabel_iter = iter(unlabed_train_loader)
    len_iter = len(unlabel_iter)
    label_iter = iter(labeled_train_loader)
    # print(len_iter) 216
    Dice = []

    if args.use_cuda:
        netS_local = netS_local.cuda()
        netC_local = netC_local.cuda()
        ema_model=ema_model.cuda()
    print('learning-rate{}'.format(lr))
    iter_num = 0
    netS_local.train()  # 学生模型初始化
    ema_model.train()  # 教师模型初始化
    optimizerG = optim.Adam(netS_local.parameters(), lr=lr, betas=(args.beta1, args.beta2))
    optimizerD = optim.Adam(netC_local.parameters(), lr=lr, betas=(args.beta1, args.beta2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(1, args.epochs + 1):
        for batch_idx in range(len_iter):
            # 梯度请0
            netC_local.zero_grad()
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

            image_s1 = image_s1.float()
            target_s1 = target_s1.float()
            image_u = image_u.float()

            if args.use_cuda:
                image_s1 = image_s1.cuda()
                target_s1 = target_s1.cuda()
                image_u = image_u.cuda()


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

            # loss_D.backward()
            # 训练分割网

            optimizerD.step()
            netS_local.zero_grad()

            output_s1 = netS_local(image_s1)
            output_s1 = F.sigmoid(output_s1)

            ul_y = netS_local(image_u)
            v_loss = vat_loss(netS_local, image_u, ul_y, xi=1e-6, eps=2.5, num_iters=1)


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

            loss_G_joint = loss_G1 + args.alpha * loss_dice+v_loss

            loss_G_joint.backward(loss_G_joint.clone().detach())
            optimizerG.step()



            if (batch_idx % 10 == 0):
                print("\nEpoch[{}/{}]\tBatch({}/{}):\tG_Loss: {:.4f}\tD_Loss: {:.4f} \n".format(
                    fed_round,  args.rounds, batch_idx,  len_iter, loss_G1.item(), loss_D.item()))
            # saving visualizations after each epoch to monitor model's progress

    return netS_local,netC_local