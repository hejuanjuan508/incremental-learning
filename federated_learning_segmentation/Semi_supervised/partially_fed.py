import copy
from torch import optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import torch
# 设置警告过滤器
import warnings
# 设置警告过滤器，忽略所有的 UserWarning
from Semi_supervised.util import cal_gradient_penalty
from tool.utils import dice_loss

warnings.filterwarnings("ignore", category=UserWarning)



def ClientUpdate(args, lr, client, netS_local, netC_local, labeled_train_loader, fed_round):
    print("This is FedAvg 客户端{}开始训练".format(client))
    if args.use_cuda:
        netS_local =copy.deepcopy(netS_local).to('cuda')
        netC_local = copy.deepcopy(netC_local).to('cuda')
    netS_local.train()

    optimizerG = optim.Adam(netS_local.parameters(), lr=lr, betas=(args.beta1, args.beta2))
    optimizerD = optim.Adam(netC_local.parameters(), lr=lr, betas=(args.beta1, args.beta2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(1, args.epochs + 1):
        for batch_idx, data in tqdm(enumerate(labeled_train_loader, 1)):

            image_s1, target_s1, gt = Variable(data[0]), Variable(data[1]), Variable(data[2])
            # 梯度请0
            netC_local.zero_grad()
            image_s1 = image_s1.float()
            target_s1 = target_s1.float()

            if args.use_cuda:
                image_s1 = image_s1.cuda()
                # image_s1=image_s1.half()
                target_s1 = target_s1.cuda()
                # target_s1=target_s1.half()

            # 有注释
            output_s1= netS_local(image_s1)
            output_s1 = torch.sigmoid(output_s1)
            output_s1 = output_s1.detach()


            # 计算多尺度损失函数

            input_mask_s1 = image_s1.clone()
            output_masked_s1 = input_mask_s1 * output_s1

            if args.use_cuda:
                output_masked_s1 = output_masked_s1.cuda()


            target_masked_s1 = input_mask_s1 * target_s1
            if args.use_cuda:
                target_masked_s1 = target_masked_s1.cuda()

            output_D1= netC_local(output_masked_s1)
            target_D1= netC_local(target_masked_s1)

            loss_D = 1-torch.mean(torch.abs(output_D1 - target_D1))

            loss_D_gp = cal_gradient_penalty(netC_local, target_s1, output_s1, device, type='mixed', constant=1.0,
                                             lambda_gp=10.0)
            loss_D_joint = loss_D+args.lambda_d*loss_D_gp

            loss_D_joint.backward()
            optimizerD.step()
            netC_local.zero_grad()

            output_s1 = netS_local(image_s1)
            output_s1 = torch.sigmoid(output_s1)
            output_masked_s1 = input_mask_s1 * output_s1
            if args.use_cuda:
                output_masked_s1 = output_masked_s1.cuda()

            target_masked_s1 = input_mask_s1 * target_s1
            if args.use_cuda:
                target_masked_s1 = target_masked_s1.cuda()

            output_G1= netC_local(output_masked_s1)
            target_G1= netC_local(target_masked_s1)
            loss_G1 = torch.mean(torch.abs(output_G1 - target_G1))
            loss_dice = dice_loss(output_s1, target_s1)
            loss_G_joint = loss_G1 + 0.1*loss_dice
            loss_G_joint.backward()
            optimizerG.step()
            netS_local.zero_grad()

            if (batch_idx % 10 == 0):
                print("\nEpoch[{}/{}]\tBatch({}/{}):\tG_Loss: {:.4f}\tD_Loss: {:.4f} \n".format(
                    fed_round, args.rounds, batch_idx, len(labeled_train_loader), loss_G_joint.item(), loss_D.item()))
        netS_local.to('cpu')
        netC_local.to('cpu')
        torch.save(netS_local.state_dict(), 'model'+str(client)+'.pth')
    return netS_local,netC_local
