import time

from tqdm import tqdm

from torch.autograd import Variable

import torch.nn.functional as F

import torch.optim as optim
import torch
import warnings

from models.net import NetS, NetC
from tool.utils import dice_loss, mergeChannels, plot_show

warnings.filterwarnings("ignore")


def create_model(ndf,ema=False):
    model =NetS(ndf)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

# calculate gradient penalty
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
        #disc_interpolates,_= netD(interpolatesv,real_data,image)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2,
                                                      dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

def local_train_dataset(args, dataset_train,dataset_test):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    ndf=32
    clients=[0,1,2,3]
    print('===> Building model\n')
    #初始化分割网络和鉴别器网络
    netS = create_model(ndf)
    netC = NetC(ndf)
    #print(netC)
    if args.use_cuda:
        netS = netS.cuda()
        netC = netC.cuda()
    #定义优化器
    lr=args.lr
    optimizerG = optim.Adam(netS.parameters(), lr=lr, betas=(args.beta1, args.beta2))
    optimizerD = optim.Adam(netC.parameters(), lr=lr, betas=(args.beta1, args.beta2))


    Dice=[]
    Dice1 = [[], [], [], []]
    print('===> Preparing data\n')
    train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=args.train_batch_size,
                                               shuffle=True,

                                               )

    val_loader = torch.utils.data.DataLoader(dataset_test,
                                             batch_size=args.val_batch_size,
                                             shuffle=False,

                                             )

    print('===> Training\n')
    size = 256
    print('learning-rate{}'.format(lr))


    netS.train()
    start_time=time.time()
    for epoch in range(1,args.epochs+1):

        for batch_idx, data in tqdm(enumerate(train_loader,1)):

            #train C


            image, target , gt = Variable(data[0]), Variable(data[1]),Variable(data[2])

            image = image.float()
            target = target.float()

            if args.use_cuda:
                image = image.cuda()
                target = target.cuda()

            output= netS(image)
            output = F.sigmoid(output)
            output = output.detach()



            #output_masked = image.clone()
            input_mask = image.clone()

            output_masked = input_mask * output
            if args.use_cuda:
                output_masked = output_masked.cuda()

            #target_masked = image.clone()
            target_masked = input_mask * target
            if args.use_cuda:
                target_masked = target_masked.cuda()

            output_D = netC(output_masked)
            target_D = netC(target_masked)

            loss_D =1-torch.mean(torch.abs(output_D - target_D))

            loss_D_gp = cal_gradient_penalty(netC, target, output,device, type='mixed', constant=1.0, lambda_gp=10.0)[0]
            loss_D_joint = loss_D + args.lambda_d * loss_D_gp

            optimizerD.zero_grad()
            loss_D_joint.backward()
            #loss_D.backward()
            optimizerD.step()

            # train G

            output = netS(image)
            output = F.sigmoid(output)

            output_masked = input_mask * output
            if args.use_cuda:
                output_masked = output_masked.cuda()
            target_masked = input_mask * target
            if args.use_cuda:
                target_masked = target_masked.cuda()

            output_G = netC(output_masked)
            target_G = netC(target_masked)

           # posi=loss_cos(output_G,target_G)

            #logits = posi.reshape(-1, 1)

            loss_dice = dice_loss(output, target)

            loss_G = torch.mean(torch.abs(output_G - target_G))


            loss_G_joint = loss_G + args.alpha * loss_dice
            # loss_G_joint = output_G.mean()
            optimizerG.zero_grad()
            loss_G_joint.backward()
            optimizerG.step()


            if(batch_idx%10==0):
                print("\nEpoch[{}/{}]\tBatch({}/{}):\tG_Loss: {:.4f}\tD_Loss: {:.4f} \n".format(
                    epoch, args.epochs, batch_idx, len(train_loader), loss_G_joint.item(), loss_D.item()))
            # saving visualizations after each epoch to monitor model's progress
        dice = test(args, netS, val_loader, size, epoch, clients[0])
        Dice.append(dice)
        print(Dice)
        if epoch % 25 == 0:
            lr = lr * args.decay_rate
            if lr <= 0.00000001:
                lr = 0.00000001
            optimizerG = optim.Adam(netS.parameters(), lr=lr, betas=(args.beta1, args.beta2))
            optimizerD = optim.Adam(netC.parameters(), lr=lr, betas=(args.beta1, args.beta2))
            print('learning-rate{}'.format(lr))
        if epoch % 10 == 0:
             for idx in range(len(clients)):
                test_loader = torch.utils.data.DataLoader(dataset_test[idx],batch_size=args.val_batch_size,shuffle=False,num_workers=args.workers)
                dice = test(args, netS, test_loader, size, epoch+100, clients[idx])
                Dice1[idx].append(dice)
                print(Dice1[idx])

    print("epoch Stopped:{:.2f}".format(time.time()-start_time))


def test(args,netS,val_loader,size,epoch,client):
    netS.eval()
    # 测试模式
    print('===>Testning\n')
    max_dice = 0
    max_iou = 0
    max_se = 0
    max_sp = 0
    max_pre = 0
    MAE_SUM = 0
    SP_SUM = 0
    for i, data in tqdm(enumerate(val_loader, 1)):

        with torch.no_grad():
            img, target, gt = Variable(data[0]), Variable(data[1]), Variable(data[2])

            if args.use_cuda:
                img = img.type(torch.FloatTensor)
                gt = gt.type(torch.FloatTensor)
                img = img.cuda()
                gt = gt.cuda()
                target = target.cuda()

            pred= netS(img)
            pred = F.sigmoid(pred)
            # print(img.shape,pred.shape)

            #pred = pred[:, 0, :, :].reshape(-1, 1, size, size)
            pred_np = torch.from_numpy(mergeChannels(pred.data.cpu().numpy(), size)).cuda()
            # print(pred_np.shape)

            N = gt.size(0)
            pred_flat = pred_np.view(N, -1)
            gt_flat = gt.view(N, -1)
            #target_flat=target.view(N, -1)
            eps = 0.0000001

            #print(pred_flat.shape,target_flat.shape)
            tn = torch.sum((1 - gt_flat) * (1 - pred_flat), dim=1)
            tp = torch.sum(gt_flat * pred_flat, dim=1)
            fp = torch.sum(pred_flat, dim=1) - tp
            fn = torch.sum(gt_flat, dim=1) - tp
            loss_iou = (tp + eps) / (tp + fp + fn + eps)
            loss_di = (2 * tp + eps) / (2 * tp + fp + fn + eps)
            loss_sp = (tn + eps) / (tn + fp + eps)
            loss_se = (tp + eps) / (tp + fn + eps)
            loss_pre = (tp + tn + eps) / (tp + tn + fp + fn + eps)
            miou = loss_iou.sum() / N
            mdice = loss_di.sum() / N
            se = loss_se.sum() / N
            sp = loss_sp.sum() / N
            pre = loss_pre.sum() / N
            MAE_loss = torch.nn.L1Loss()
            MAE = MAE_loss(pred_flat, gt_flat)

    netS.train()

    print(
        '-------------------------------------------------------------------------------------------------------------------\n')

    MAE_SUM = MAE_SUM + MAE
    mMAE = MAE_SUM / args.epochs
    SP_SUM = SP_SUM + sp
    mSP = SP_SUM / args.epochs
    # IoUs = np.array(IoUs, dtype=np.float64)
    # dices = np.array(dices, dtype=np.float64)
    # mIoU = np.nanmean(IoUs, axis=0)
    # mdice = np.nanmean(dices, axis=0)
    print('mIoU: {:.4f}'.format(miou))
    print('Dice: {:.4f}'.format(mdice))
    print('SE: {:.4f}'.format(se))
    print('SP: {:.4f}'.format(sp))
    print('Pre:{:.4f}'.format(pre))
    print('MAE: {:.4f}'.format(MAE))

    # print('SM: {:.4f}'.format(sm))

    if max_se < se < 0.99:
        max_se = se
    if max_sp < sp < 0.99:
        max_sp = sp
    if max_iou < miou:
        max_iou = miou
    if max_dice < mdice:
        max_dice = mdice
    if max_pre < pre:
        max_dice = pre

    plot_show(epoch, client, img.data.cpu(), gt.data.cpu(), pred.data.cpu(), mdice.data.cpu(), flag=0)

    # visualizeMasks(epoch,fed_round,client, img.data.cpu(), gt.data.cpu(), target.data.cpu(), pred.data.cpu(), size=256, flag=0)
    if epoch % 25 == 0:

        # print('K: {:.4f}'.format(k))
        print('MAE: {:.4f}'.format(mMAE))
        print('SP: {:.4f}'.format(mSP))
        print('MaxSE: {:.4f}  '.format(max_se))
        print('MaxSP: {:.4f} '.format(max_sp))
        print('MaxPre: {:.4f} '.format(max_pre))
        print('MaxmIoU: {:.4f}'.format(max_iou))
        print('MaxDice: {:.4f} '.format(max_dice))
        torch.save(netS.state_dict(), '%s/checkpoint_epoch_%d.pth' % (args.outpath, args.epochs))
    torch.cuda.empty_cache()

    print(
        '-------------------------------------------------------------------------------------------------------------------')
    #plot_loss(epoch_lossD,epoch_lossG)
