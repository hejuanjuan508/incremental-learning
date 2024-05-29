import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable





#用全局模型进行预测
from tool.utils import mergeChannels, plot_show


def  fed_test(args,server_model,val_loader,fed_round,client,flag):

    if args.use_cuda:
        server_model = server_model.cuda()

    print(
        '-------------------------------------------------------------------------------------------------------------------\n')
    #测试一下模型的大小
    torch.save(server_model, 'global.pth')
    #print("客户端{}开始测试".format(client))
    #print(global_model)
    server_model.eval()
    max_dice = 0
    max_iou = 0
    max_se = 0
    max_sp = 0
    MAE_SUM = 0
    SP_SUM = 0
    for i, data in tqdm(enumerate(val_loader, 1)):

        with torch.no_grad():
            img, target, gt= Variable(data[0]), Variable(data[1]), Variable(data[2])

            if args.use_cuda:
                img = img.type(torch.FloatTensor)
                gt = gt.type(torch.FloatTensor)
                img = img.cuda()
                gt = gt.cuda()
                target = target.cuda()
            else:
                img = img.type(torch.FloatTensor)
                gt = gt.type(torch.FloatTensor)


            pred= server_model(img)
            pred = F.sigmoid(pred)
            #pred = pred[:, 0, :, :].view(-1, 1, size, size)
            # print(img.shape,pred.shape)

            # pred = pred[:, 0, :, :].reshape(-1, 1, size, size)

            pred_np = torch.from_numpy(mergeChannels(pred.data.cpu().numpy(), img.cpu().size(2))).cuda()
            # print(pred_np.shape)

            N = gt.size(0)
            pred_flat = pred_np.view(N, -1)
            gt_flat = gt.view(N, -1)
            eps = 0.0000001

            tn = torch.sum((1 - gt_flat) * (1 - pred_flat), dim=1)
            tp = torch.sum(gt_flat * pred_flat, dim=1)
            fp = torch.sum(pred_flat, dim=1) - tp
            fn = torch.sum(gt_flat, dim=1) - tp
            loss_iou = (tp + eps) / (tp + fp + fn + eps)
            loss_di = (2 * tp + eps) / (2 * tp + fp + fn + eps)
            loss_sp = tn / (tn + fp)
            loss_se = (tp + eps) / (tp + fn + eps)
            loss_pre = (tp + tn + eps) / (tp + tn + fp + fn + eps)
            miou = loss_iou.sum() / N
            mdice = loss_di.sum() / N
            sp = loss_sp.sum() / N
            pre = loss_pre.sum() / N
            se = loss_se.sum() / N
            MAE_loss = torch.nn.L1Loss()
            MAE = MAE_loss(pred_flat, gt_flat)


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
    print('SP: {:.4f}'.format(sp))
    print('Pre:{:.4f}'.format(pre))
    print('SE: {:.4f}'.format(se))
    print('MAE: {:.4f}'.format(MAE))
    #print('SM: {:.4f}'.format(sm))



    if max_se < se < 0.99:
        max_se = se
    if max_sp < sp < 0.99:
        max_sp = sp
    if max_iou < miou:
        max_iou = miou
    if max_dice < mdice:
        max_dice = mdice


    plot_show(fed_round,client, img.data.cpu(), gt.data.cpu(), pred.data.cpu(),mdice.data.cpu(),flag)

    #visualizeMasks(fed_round,fed_round,client, img.data.cpu(), gt.data.cpu(), target.data.cpu(), pred.data.cpu(), size=256,flag=0)

    if fed_round % 25 == 0:

        print('MAE: {:.4f}'.format(mMAE))
        print('SP: {:.4f}'.format(mSP))
        print('MaxSE: {:.4f} '.format(max_se))
        print('MaxSP: {:.4f} '.format(max_sp))
        print('MaxmIoU: {:.4f}'.format(max_iou))
        print('MaxDice: {:.4f}'.format(max_dice))
       # torch.save(netS.state_dict(), '%s/checkpoint_epoch_%d.pth' % (args.outpath, args.epochs))
    torch.cuda.empty_cache()
    server_model.to('cpu')

    return mdice
