import copy

from torch.cuda.amp import autocast, GradScaler
from torch import optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from tqdm import tqdm
import numpy as np
import torch.autograd as autograd

import torch
# 设置警告过滤器
import warnings
# 设置警告过滤器，忽略所有的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)


autograd.set_detect_anomaly(True)

from tool.utils import dice_loss, mergeChannels, CriterionKD


# def quant_fx(model):
#     """
#     使用Pytorch中的FX模式对模型进行量化
#     """
#     model.eval()
#     qconfig = get_default_qconfig("fbgemm")  # 默认是静态量化
#     qconfig_dict = {
#         "": qconfig,
#         # 'object_type': []
#     }
#     model_to_quantize = copy.deepcopy(model)
#     prepared_model = prepare_fx(model_to_quantize, qconfig_dict)
#     quantized_model = convert_fx(prepared_model)
#     torch.save(quantized_model.state_dict(), "r18_quant.pth")
#     return quantized_model

def soft_threshold(w, th):
	'''
	pytorch soft-sign function
	'''
	with torch.no_grad():
		temp = torch.abs(w) - th
		# print('th:', th)
		# print('temp:', temp.size())
		return torch.sign(w) * nn.functional.relu(temp)




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

def cross_entropy2d(logit, target, ignore_index=255, weight=None, size_average=True, batch_average=True):
    n, c, h, w = logit.size()
    # logit = logit.permute(0, 2, 3, 1)
    #logit=logit.squeeze(1)
    target = target.squeeze(1)
    if weight is None:
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, size_average=False)
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float().cuda(), ignore_index=ignore_index, size_average=False)
    loss = criterion(logit, target)

    if size_average:
        loss /= (h * w)

    if batch_average:
        loss /= n

    return loss

def cal_gradient_penalty(netD,real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0],
                                 real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)

        disc_interpolates= netD(interpolatesv)
        gradients_ = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True,retain_graph=True, only_inputs=True,allow_unused=True)
        gradients_ = gradients_[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients_ + 1e-16).norm(2,
                                                      dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients_
    else:
        return 0.0, None
def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def MixFedGAN_ClientUpdate(args, lr, client, netS_local, netC_local, meme, labeled_train_loader,val_loader, fed_round):
    print("This is MixFedAGAN 客户端{}开始训练".format(client))
    # qconfig_dict = {"": torch.quantization.get_default_qconfig('fbgemm')}
    # # Prepare model
    # model_prepared = quantize_fx.prepare_qat_fx(copy.deepcopy(netS_local), qconfig_dict)

    #iter_num = 0
    #loss_function = nn.BCELoss()
    criterion_kd = CriterionKD().cuda()
    #criterion_kd=CriterionPairWiseforWholeFeatAfterPool().cuda()

    # 将自定义的量化配置添加到字典中
    # 设置量化配置
    # qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    # netS_local.qconfig = qconfig
    # # 准备模型以启用 QAT
    # torch.quantization.prepare_qat(netS_local, inplace=True)

    if args.use_cuda:
        netS_local =copy.deepcopy(netS_local).to('cuda')
        netC_local = copy.deepcopy(netC_local).to('cuda')
        meme = meme.cuda()

    netS_local.train()
    #optimizerG = optim.SGD(netS_local.parameters(), lr=0.01, momentum=0.9)
    #optimizerD = optim.SGD(netC_local.parameters(), lr=0.01, momentum=0.9)
    optimizerG = optim.Adam(netS_local.parameters(), lr=lr, betas=(args.beta1, args.beta2))
    optimizerD = optim.Adam(netC_local.parameters(), lr=lr, betas=(args.beta1, args.beta2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建 GradScaler 对象
    scaler = GradScaler()
    for epoch in range(1, args.epochs + 1):
        for batch_idx, data in tqdm(enumerate(labeled_train_loader, 1)):
            # 梯度请0
            netC_local.zero_grad()
            image_s1, target_s1, gt = Variable(data[0]), Variable(data[1]), Variable(data[2])
            # 梯度请0
            netC_local.zero_grad()
            image_s1 = image_s1.float()
            target_s1 = target_s1.float()
            #image_u = image_u.float()


            if args.use_cuda:
                #new_data = new_data.float().cuda()
                image_s1 = image_s1.cuda()
                target_s1 = target_s1.cuda()
                #image_u = image_u.cuda()

            use_amp = False
            # 前向过程(model + loss)开启 autocast
            with autocast(enabled=use_amp):
            # 有注释
                output_s1= netS_local(image_s1)
            #output_s1 = logits[:args.train_batch_size]
                output_s1 = torch.sigmoid(output_s1)
                output_s1 = output_s1.detach()


            # 计算多尺度损失函数
            # output_masked = image.clone()
                input_mask_s1 = image_s1.clone()
            #input_mask_u = image_u.clone()
                output_masked_s1 = input_mask_s1 * output_s1

                if args.use_cuda:
                    output_masked_s1 = output_masked_s1.cuda()

            # target_masked = image.clone()
                target_masked_s1 = input_mask_s1 * target_s1
                if args.use_cuda:
                    target_masked_s1 = target_masked_s1.cuda()

            # 将注释的图像视为真
            # print(output_masked_s1.shape, output_s1.shape, image_s1.shape)
                output_D1= netC_local(output_masked_s1)

            # print(pred_real.shape, valid.shape)

                target_D1= netC_local(target_masked_s1)

                loss_D = 1 - torch.mean(torch.abs(output_D1 - target_D1))

                loss_D_gp = cal_gradient_penalty(netC_local, target_s1, output_s1, device, type='mixed', constant=1.0,
                                             lambda_gp=10.0)[0]
                loss_D_joint = loss_D+args.lambda_d*loss_D_gp

            scaler.scale(loss_D_joint).backward()
            # 反向传播在autocast上下文之外
            scaler.step(optimizerD)
            scaler.update()
            netC_local.zero_grad()
            # 训练分割网
            use_amp = False
            with autocast(enabled=use_amp):
                netS_local.zero_grad()

                output_s1 = netS_local(image_s1)
            #output_s1 = logits[:args.train_batch_size]
                output_s1 = torch.sigmoid(output_s1)

                with torch.no_grad():
                    output_s2 = meme(image_s1)
                #output_s2 = logits1[:args.train_batch_size]
                    output_s2 = torch.sigmoid(output_s2)

                kl_local=criterion_kd(output_s1, output_s2)

            # 计算多尺度L1损失函数
            # output_masked = image.clone()
                output_masked_s1 = input_mask_s1 * output_s1
            #output_masked_s2 = input_mask_u * output_s2
                if args.use_cuda:
                    output_masked_s1 = output_masked_s1.cuda()

            # target_masked = image.clone()
                target_masked_s1 = input_mask_s1 * target_s1
                if args.use_cuda:
                    target_masked_s1 = target_masked_s1.cuda()

                output_G1= netC_local(output_masked_s1)
                target_G1= netC_local(target_masked_s1)
                loss_G1 = torch.mean(torch.abs(output_G1 - target_G1))
                loss_dice = dice_loss(output_s1, target_s1)
            # loss_dice1 = dice_loss(output_s2, target_s1)

                loss_G_joint = loss_G1 + args.alpha * loss_dice+0.1*kl_local
            scaler.scale(loss_G_joint).backward()
            scaler.step(optimizerG)
            scaler.update()
            # teacher模型更新为当前student和上一个epoch的teacher模型的加权，即EMA平滑版本。
            # update_ema_variables(netS, meme, args.ema_decay, iter_num)
            #iter_num = iter_num + 1

            if (batch_idx % 10 == 0):
                print("\nEpoch[{}/{}]\tBatch({}/{}):\tG_Loss: {:.4f}\tD_Loss: {:.4f} \n".format(
                    fed_round, args.rounds, batch_idx, len(labeled_train_loader), loss_G_joint.item(), loss_D.item()))

        dice = test(args,netS_local, val_loader)
        #dice1 = test(args,meme, val_loader)

        # 在剪枝之前记录模型参数数量
        # proximal gradient for channel pruning:


        # # proximal gradient for channel pruning:
        # for name, m in netC_local.named_modules():
        #     if isinstance(m, nn.BatchNorm2d) and m.weight is not None:
        #         m.weight.data = soft_threshold(m.weight.data, th=float(args.rho) * float(lr))
        # 在剪枝之后记录模型参数数量



        netS_local.to('cpu')
        netC_local.to('cpu')
        meme.to('cpu')


        params1 = list(meme.parameters())
        params2 = list(netS_local.parameters())
        distance = 0.0
        for w, w_t in zip(params1,params2):
            distance += torch.norm(w - w_t)
        distance /= len(params1)  # 求平均距离
        # 归一化确保相似性在[0,1]之间
        similarity = 1.0 / (1.0 + distance)  # 相似性度量

        # # 获取模型的量化配置
        # model_quantized.load_state_dict(model_para)
        #
        # # 获取模型的量化配置
        # qconfig = torch.quantization.get_default_qconfig('fbgemm')
        # # 检查激活值（activation）和权重（weight）的位数
        # # 获取激活值的位数
        # activation_bits = qconfig.activation().dtype
        # weight_bits = qconfig.weight().dtype
        # print(f"激活值位数: {activation_bits}")
        # print(f"权重位数: {weight_bits}")


        # # 在逆量化后，需要将量化参数设置回来
        # for name, param in model_quantized.named_parameters():
        #     model_quantized.register_parameter(name, param.dequantize())
        # Convert model to int8
        # print('Converting model to int8...')
        # model_quantized = quantize_fx.convert_fx(netS_local)  # 将prepared模型转换成真正的int8定点模型


    return netS_local,netC_local,dice,similarity
