import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt

from thop import profile
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

classes = 2
channel_dim = 1
norm_mean, norm_var = 0.0, 1.0
_NBITS = 4
_ACTMAX = 4.0
NUM_BITS = 8
NUM_BITS_WEIGHT = 8
NUM_BITS_GRAD = 8
BIPRECISION = True
ndf = 32
class channel_selection(nn.Module):
    """
    Select channels from the output of BatchNorm2d layer. It should be put directly after BatchNorm2d layer.
    The output shape of this layer is determined by the number of 1 in `self.indexes`.
    """
    def __init__(self, num_channels):
        """
        Initialize the `indexes` with all one vector with the length same as the number of channels.
        During pruning, the places in `indexes` which correpond to the channels to be pruned will be set to 0.
        """
        super(channel_selection, self).__init__()
        self.indexes = nn.Parameter(torch.ones(num_channels))

    def forward(self, input_tensor):
        """
        Parameter
        ---------
        input_tensor: (N,C,H,W). It should be the output of BatchNorm2d layer.
        """
        selected_index = np.squeeze(np.argwhere(self.indexes.data.cpu().numpy()))
        if selected_index.size == 1:
            selected_index = np.resize(selected_index, (1,))
        output = input_tensor[:, selected_index, :, :]
        return output

#整个模型进行通道剪枝
def channel_pruning_unet(args,model,percentage):
    channel_importance = {}
    channel_importance1 = {}

    # 获取模型的命名子模块
    # named_modules = dict(model.named_modules())
    for (name, module) in  model.named_modules():

        if isinstance(module, nn.Conv2d) and name != 'deconvblock9.0':
            # print(name)

            if args.compress_mode == 'l2':
                channel_importance[name] = torch.norm(module.weight.data, 2, dim=(1, 2, 3))
                # 函数对这个一维数组进行排序，返回按照从小到大排序后的索引数组。这个索引数组表示了通道重要性从低到高的顺序。
                sorted_channels = np.argsort(
                    np.concatenate([x.cpu().numpy().flatten() for x in channel_importance[name]]))
                num_channels_to_prune = int(module.weight.data.size(0) * (1 - percentage))
                # print(sorted_channels[:num_channels_to_prune])
                # print('通道总的通道数', sorted_channels)
                # 对权重矩阵进行通道剪枝
                pruned_channel_indices = sorted_channels[num_channels_to_prune:]
                # print('通道留下的通道数', len(pruned_channel_indices))

                channel_importance1[name] = torch.norm(module.weight.data, 2, dim=(0, 2, 3))
                # 函数对这个一维数组进行排序，返回按照从小到大排序后的索引数组。这个索引数组表示了通道重要性从低到高的顺序。
                sorted_channels1 = np.argsort(
                    np.concatenate([x.cpu().numpy().flatten() for x in channel_importance1[name]]))
                num_channels_to_prune1 = int(module.weight.data.size(1) * (1 - percentage))
                pruned_channel_indices1 = sorted_channels1[num_channels_to_prune1:]
                # print('通道1留下的通道数', pruned_channel_indices)

            pruned_channel_indices = torch.tensor(pruned_channel_indices)
            # pruned_channel_indices = pruned_channel_indices.clone().detach()
            pruned_channel_indices = pruned_channel_indices.to(module.weight.device)
            module.weight.data = torch.index_select(module.weight.data, 0, pruned_channel_indices)

            pruned_channel_indices1 = torch.tensor(pruned_channel_indices1)
            # pruned_channel_indices1 = pruned_channel_indices1.clone().detach()
            pruned_channel_indices1 = pruned_channel_indices1.to(module.weight.device)
            module.weight.data = torch.index_select(module.weight.data, 1, pruned_channel_indices1)

            # print(pruned_module.weight.data.shape)
            if module.bias is not None:

                module.bias.data = torch.index_select(module.bias.data, 0, pruned_channel_indices)

        if (name == 'deconvblock9.0'):
            # if module.bias is not None:
            #     pruned_module.bias.data = module.bias.data[sorted_channels[num_channels_to_prune:]]
            if args.compress_mode == 'l2':

                channel_importance1[name] = torch.norm(module.weight.data, 2, dim=(0, 2, 3))
                # 函数对这个一维数组进行排序，返回按照从小到大排序后的索引数组。这个索引数组表示了通道重要性从低到高的顺序。
                sorted_channels1 = np.argsort(
                    np.concatenate([x.cpu().numpy().flatten() for x in channel_importance1[name]]))
                num_channels_to_prune1 = int(module.weight.data.size(1) * (1 - percentage))
                # print('通道1总的通道数', sorted_channels)
                pruned_channel_indices1 = sorted_channels1[num_channels_to_prune1:]
            module.weight.data = module.weight.data[:, pruned_channel_indices1]
            # print(pruned_module.weight.data.shape)
            if module.bias is not None:
                module.bias.data = torch.index_select(module.bias.data, 0, pruned_channel_indices1)

        elif isinstance(module, nn.BatchNorm2d):
            pruned_channel_indices = pruned_channel_indices.to(module.weight.device)
            module.weight.data = torch.index_select(module.weight.data, 0, pruned_channel_indices)
            if module.bias is not None:
                module.bias.data = torch.index_select(module.bias.data, 0, pruned_channel_indices)

            if module.running_mean is not None:
                module.running_mean.data = torch.index_select(module.running_mean.data, 0,
                                                                     pruned_channel_indices)
            if module.running_var is not None:
                module.running_var.data = torch.index_select(module.running_var.data, 0,
                                                                    pruned_channel_indices)
    return model


def channel_pruning_unet1(args,model,percentage,ndf,percentage1):
    channel_importance = {}
    channel_importance1 = {}
    new_ndf=(int)(ndf*percentage)
    new_ndf1 = (int)(ndf * percentage1)
    pruned_model=NetS(new_ndf)

    # 获取模型的命名子模块
    # named_modules = dict(model.named_modules())
    inner_layer={'convblock6_1.0','convblock7.0','convblock8.0','deconvblock1.0','deconvblock2.0','deconvblock2_1.0'}
    for (name, module), (name1, pruned_module) in zip(model.named_modules(), pruned_model.named_modules()):
        if isinstance(module, nn.Conv2d) and (name != 'deconvblock9.0') and (name not in inner_layer )  :

            if args.compress_mode == 'l2':
                channel_importance[name] = torch.norm(module.weight.data, 2, dim=(1, 2, 3))
                # 函数对这个一维数组进行排序，返回按照从小到大排序后的索引数组。这个索引数组表示了通道重要性从低到高的顺序。
                sorted_channels = np.argsort(
                    np.concatenate([x.cpu().numpy().flatten() for x in channel_importance[name]]))
                num_channels_to_prune = int(module.weight.data.size(0) * (1 - percentage))
                # print(sorted_channels[:num_channels_to_prune])
                # print('通道总的通道数', sorted_channels)
                # 对权重矩阵进行通道剪枝
                pruned_channel_indices = sorted_channels[num_channels_to_prune:]
                # print('通道留下的通道数', len(pruned_channel_indices))

                channel_importance1[name] = torch.norm(module.weight.data, 2, dim=(0, 2, 3))
                # 函数对这个一维数组进行排序，返回按照从小到大排序后的索引数组。这个索引数组表示了通道重要性从低到高的顺序。
                sorted_channels1 = np.argsort(
                    np.concatenate([x.cpu().numpy().flatten() for x in channel_importance1[name]]))
                num_channels_to_prune1 = int(module.weight.data.size(1) * (1 - percentage))
                # print('通道1总的通道数', sorted_channels)
                pruned_channel_indices1 = sorted_channels1[num_channels_to_prune1:]
                # print('通道1留下的通道数', pruned_channel_indices)

            pruned_channel_indices = torch.tensor(pruned_channel_indices)
            # pruned_channel_indices = pruned_channel_indices.clone().detach()
            pruned_channel_indices = pruned_channel_indices.to(module.weight.device)
            pruned_module.weight.data = torch.index_select(module.weight.data, 0, pruned_channel_indices)

            pruned_channel_indices1 = torch.tensor(pruned_channel_indices1)
            # pruned_channel_indices1 = pruned_channel_indices1.clone().detach()
            pruned_channel_indices1 = pruned_channel_indices1.to(module.weight.device)
            pruned_module.weight.data = torch.index_select(pruned_module.weight.data, 1, pruned_channel_indices1)

            # print(pruned_module.weight.data.shape)
            if module.bias is not None:

                pruned_module.bias.data = torch.index_select(module.bias.data, 0, pruned_channel_indices)


        if (name == 'convblock6.0'):


            if args.compress_mode == 'l2':
                channel_importance[name] = torch.norm(module.weight.data, 2, dim=(1, 2, 3))
                # 函数对这个一维数组进行排序，返回按照从小到大排序后的索引数组。这个索引数组表示了通道重要性从低到高的顺序。
                sorted_channels = np.argsort(
                    np.concatenate([x.cpu().numpy().flatten() for x in channel_importance[name]]))
                num_channels_to_prune = int(module.weight.data.size(0) * (1 - percentage1))
                # print(sorted_channels[:num_channels_to_prune])
                # print('通道总的通道数', sorted_channels)
                # 对权重矩阵进行通道剪枝
                pruned_channel_indices = sorted_channels[num_channels_to_prune:]
                # print('通道留下的通道数', pruned_channel_indices)

                channel_importance1[name] = torch.norm(module.weight.data, 2, dim=(0, 2, 3))
                # 函数对这个一维数组进行排序，返回按照从小到大排序后的索引数组。这个索引数组表示了通道重要性从低到高的顺序。
                sorted_channels1 = np.argsort(
                    np.concatenate([x.cpu().numpy().flatten() for x in channel_importance1[name]]))
                num_channels_to_prune1 = int(module.weight.data.size(1) * (1 - percentage))
                # print('通道1总的通道数', sorted_channels)
                pruned_channel_indices1 = sorted_channels1[num_channels_to_prune1:]

            pruned_channel_indices = torch.tensor(pruned_channel_indices)
            # pruned_channel_indices = pruned_channel_indices.clone().detach()
            pruned_channel_indices = pruned_channel_indices.to(module.weight.device)
            pruned_module.weight.data = torch.index_select(module.weight.data, 0, pruned_channel_indices)

            pruned_channel_indices1 = torch.tensor(pruned_channel_indices1)
            # pruned_channel_indices1 = pruned_channel_indices1.clone().detach()
            pruned_channel_indices1 = pruned_channel_indices1.to(module.weight.device)
            pruned_module.weight.data = torch.index_select(pruned_module.weight.data, 1, pruned_channel_indices1)

            # print(pruned_module.weight.data.shape)
            if module.bias is not None:
                pruned_module.bias.data = torch.index_select(module.bias.data, 0, pruned_channel_indices)


        if(name in inner_layer):

            if args.compress_mode == 'l2':
                channel_importance[name] = torch.norm(module.weight.data, 2, dim=(1, 2, 3))
                # 函数对这个一维数组进行排序，返回按照从小到大排序后的索引数组。这个索引数组表示了通道重要性从低到高的顺序。
                sorted_channels = np.argsort(
                    np.concatenate([x.cpu().numpy().flatten() for x in channel_importance[name]]))
                num_channels_to_prune = int(module.weight.data.size(0) * (1 - percentage1))
                # print(sorted_channels[:num_channels_to_prune])
                # print('通道总的通道数', sorted_channels)
                # 对权重矩阵进行通道剪枝
                pruned_channel_indices = sorted_channels[num_channels_to_prune:]
                # print('通道留下的通道数', pruned_channel_indices)

                channel_importance1[name] = torch.norm(module.weight.data, 2, dim=(0, 2, 3))
                # 函数对这个一维数组进行排序，返回按照从小到大排序后的索引数组。这个索引数组表示了通道重要性从低到高的顺序。
                sorted_channels1 = np.argsort(
                    np.concatenate([x.cpu().numpy().flatten() for x in channel_importance1[name]]))
                num_channels_to_prune1 = int(module.weight.data.size(1) * (1 - percentage1))
                # print('通道1总的通道数', sorted_channels)
                pruned_channel_indices1 = sorted_channels1[num_channels_to_prune1:]
                # print('通道1留下的通道数', pruned_channel_indices)

            pruned_channel_indices = torch.tensor(pruned_channel_indices)
            # pruned_channel_indices = pruned_channel_indices.clone().detach()
            pruned_channel_indices = pruned_channel_indices.to(module.weight.device)
            pruned_module.weight.data = torch.index_select(module.weight.data, 0, pruned_channel_indices)

            pruned_channel_indices1 = torch.tensor(pruned_channel_indices1)
            # pruned_channel_indices1 = pruned_channel_indices1.clone().detach()
            pruned_channel_indices1 = pruned_channel_indices1.to(module.weight.device)
            pruned_module.weight.data = torch.index_select(pruned_module.weight.data, 1, pruned_channel_indices1)


            # print(pruned_module.weight.data.shape)
            if module.bias is not None:
                pruned_module.bias.data = torch.index_select(module.bias.data, 0, pruned_channel_indices)


        if (name == 'deconvblock3.0'):

            if args.compress_mode == 'l2':
                channel_importance[name] = torch.norm(module.weight.data, 2, dim=(1, 2, 3))
                # 函数对这个一维数组进行排序，返回按照从小到大排序后的索引数组。这个索引数组表示了通道重要性从低到高的顺序。
                sorted_channels = np.argsort(
                    np.concatenate([x.cpu().numpy().flatten() for x in channel_importance[name]]))
                num_channels_to_prune = int(module.weight.data.size(0) * (1 - percentage))
                # print(sorted_channels[:num_channels_to_prune])
                # print('通道总的通道数', sorted_channels)
                # 对权重矩阵进行通道剪枝
                pruned_channel_indices = sorted_channels[num_channels_to_prune:]
                # print('通道留下的通道数', pruned_channel_indices)

                channel_importance1[name] = torch.norm(module.weight.data, 2, dim=(0, 2, 3))
                # 函数对这个一维数组进行排序，返回按照从小到大排序后的索引数组。这个索引数组表示了通道重要性从低到高的顺序。
                sorted_channels1 = np.argsort(
                    np.concatenate([x.cpu().numpy().flatten() for x in channel_importance1[name]]))
                num_channels_to_prune1 = int(module.weight.data.size(1) * (1 - percentage1))
                # print('通道1总的通道数', sorted_channels)
                pruned_channel_indices1 = sorted_channels1[num_channels_to_prune1:]

            pruned_channel_indices = torch.tensor(pruned_channel_indices)
            # pruned_channel_indices = pruned_channel_indices.clone().detach()
            pruned_channel_indices = pruned_channel_indices.to(module.weight.device)
            pruned_module.weight.data = torch.index_select(module.weight.data, 0, pruned_channel_indices)

            pruned_channel_indices1 = torch.tensor(pruned_channel_indices1)
            # pruned_channel_indices1 = pruned_channel_indices1.clone().detach()
            pruned_channel_indices1 = pruned_channel_indices1.to(module.weight.device)
            pruned_module.weight.data = torch.index_select(pruned_module.weight.data, 1, pruned_channel_indices1)

            # print(pruned_module.weight.data.shape)
            if module.bias is not None:
                pruned_module.bias.data = torch.index_select(module.bias.data, 0, pruned_channel_indices)

        if (name == 'deconvblock9.0'):
            # if module.bias is not None:
            #     pruned_module.bias.data = module.bias.data[sorted_channels[num_channels_to_prune:]]
            if args.compress_mode == 'l2':

                channel_importance1[name] = torch.norm(module.weight.data, 2, dim=(0, 2, 3))
                # 函数对这个一维数组进行排序，返回按照从小到大排序后的索引数组。这个索引数组表示了通道重要性从低到高的顺序。
                sorted_channels1 = np.argsort(
                    np.concatenate([x.cpu().numpy().flatten() for x in channel_importance1[name]]))
                num_channels_to_prune1 = int(module.weight.data.size(1) * (1 - percentage))
                # print('通道1总的通道数', sorted_channels)
                pruned_channel_indices1 = sorted_channels1[num_channels_to_prune1:]
            pruned_module.weight.data = module.weight.data[:, pruned_channel_indices1]
            # print(pruned_module.weight.data.shape)
            if module.bias is not None:
                pruned_module.bias.data = torch.index_select(module.bias.data, 0, pruned_channel_indices1)

        if isinstance(module, nn.BatchNorm2d):

            pruned_channel_indices = pruned_channel_indices.to(module.weight.device)

            pruned_module.weight.data = torch.index_select(module.weight.data, 0, pruned_channel_indices)
            if module.bias is not None:
                pruned_module.bias.data = torch.index_select(module.bias.data, 0, pruned_channel_indices)

            if module.running_mean is not None:
                pruned_module.running_mean.data = torch.index_select(module.running_mean.data, 0,
                                                                     pruned_channel_indices)
            if module.running_var is not None:
                pruned_module.running_var.data = torch.index_select(module.running_var.data, 0,
                                                                    pruned_channel_indices)
    return pruned_model

# 对编码器和解码器进行修剪
def prune_convblock_l2(model, percentage):
    channel_importance = {}
    channel_importance1 = {}
    pruned_channel_indices = None  # 初始化为 None
    for (name, module) in model.named_modules():
        if (name == 'convblock1.0'):
            print(name)
            print("Original Weight Shape:", module.weight.data.shape)
            channel_importance[name] = torch.norm(module.weight.data, 2, dim=(1, 2, 3))
            # 函数对这个一维数组进行排序，返回按照从小到大排序后的索引数组。这个索引数组表示了通道重要性从低到高的顺序。
            sorted_channels1 = np.argsort(
                np.concatenate([x.cpu().numpy().flatten() for x in channel_importance[name]]))
            num_channels_to_prune1 = int(module.weight.data.size(0) * (1 - percentage))
            # print('通道1总的通道数', sorted_channels)
            pruned_channel_indices = sorted_channels1[:num_channels_to_prune1]
            pruned_channel_indices = torch.tensor(pruned_channel_indices)
            pruned_channel_indices = pruned_channel_indices.to(module.weight.device)

            module.weight.data = torch.index_select(module.weight.data, 0, pruned_channel_indices)
            # print(pruned_module.weight.data.shape)
            print("Num Channels After Pruning:", module.weight.data.shape)

            if module.bias is not None:
                module.bias.data = torch.index_select(module.bias.data, 0, pruned_channel_indices)


        if isinstance(module, nn.Conv2d) and "deconvblock" not in name and name!=('convblock1.0') and name!=('convblock8.0'):
            print(name)
            print("Original Weight Shape:", module.weight.data.shape)

            # 计算通道重要性
            channel_importance[name] = torch.norm(module.weight.data, 2, dim=(1, 2, 3))

            sorted_channels = np.argsort(np.concatenate([x.cpu().numpy().flatten() for x in channel_importance[name]]))
            num_channels_to_prune = int(module.weight.data.size(0) * percentage)

            pruned_channel_indices = sorted_channels[:num_channels_to_prune]
            pruned_channel_indices = torch.tensor(pruned_channel_indices)
            pruned_channel_indices = pruned_channel_indices.to(module.weight.device)

            channel_importance1[name] = torch.norm(module.weight.data, 2, dim=(0, 2, 3))
            # 函数对这个一维数组进行排序，返回按照从小到大排序后的索引数组。这个索引数组表示了通道重要性从低到高的顺序。
            sorted_channels1 = np.argsort(
                np.concatenate([x.cpu().numpy().flatten() for x in channel_importance1[name]]))
            num_channels_to_prune1 = int(module.weight.data.size(1) * (1 - percentage))
            # print('通道1总的通道数', sorted_channels)
            pruned_channel_indices1 = sorted_channels1[:num_channels_to_prune1]
            pruned_channel_indices1 = torch.tensor(pruned_channel_indices1)
            pruned_channel_indices1 = pruned_channel_indices1.to(module.weight.device)

            # 执行剪枝
            module.weight.data = torch.index_select(module.weight.data, 0, pruned_channel_indices)
            print(module.weight.data.shape)

            module.weight.data = torch.index_select(module.weight.data, 1, pruned_channel_indices1)

            print("Num Channels After Pruning:", module.weight.data.shape)

            if module.bias is not None:
                module.bias.data = torch.index_select(module.bias.data, 0, pruned_channel_indices)

        if (name == 'convblock8.0'):
            print(name)
            print("Original Weight Shape:", module.weight.data.shape)
            channel_importance[name] = torch.norm(module.weight.data, 2, dim=(0, 2, 3))
            # 函数对这个一维数组进行排序，返回按照从小到大排序后的索引数组。这个索引数组表示了通道重要性从低到高的顺序。
            sorted_channels1 = np.argsort(
                np.concatenate([x.cpu().numpy().flatten() for x in channel_importance[name]]))
            num_channels_to_prune1 = int(module.weight.data.size(1) * (1 - percentage))
            # print('通道1总的通道数', sorted_channels)
            pruned_channel_indices1 = sorted_channels1[num_channels_to_prune1:]
            module.weight.data = module.weight.data[:, pruned_channel_indices1]
            print("Num Channels After Pruning:", module.weight.data.shape)
            if module.bias is not None:
                module.bias.data = torch.index_select(module.bias.data, 0, pruned_channel_indices1)

        if isinstance(module, nn.Conv2d) and "deconvblock2.0"in name:
            print(name)
            print(module.weight.data.shape)
            print("Original Weight Shape:", module.weight.data.shape)
            channel_importance[name] = torch.norm(module.weight.data, 2, dim=(0, 2, 3))
            # 函数对这个一维数组进行排序，返回按照从小到大排序后的索引数组。这个索引数组表示了通道重要性从低到高的顺序。
            sorted_channels1 = np.argsort(
                np.concatenate([x.cpu().numpy().flatten() for x in channel_importance[name]]))
            num_channels_to_prune1 = int(module.weight.data.size(1) * (1 - 0.25))

            pruned_channel_indices = sorted_channels1[:num_channels_to_prune1]
            pruned_channel_indices = torch.tensor(pruned_channel_indices)
            pruned_channel_indices = pruned_channel_indices.to(module.weight.device)
            # 执行剪枝
            print('通道1总的通道数', len(pruned_channel_indices))
            module.weight.data = torch.index_select(module.weight.data,1, pruned_channel_indices)

            print("Num Channels After Pruning:", module.weight.data.shape)
            if module.bias is not None:
                module.bias.data = torch.index_select(module.bias.data, 0, pruned_channel_indices)


        if isinstance(module, nn.BatchNorm2d) and "deconvblock" not in name and name!='convblock8.1' :
            print(name)
            print('1')

            pruned_channel_indices = pruned_channel_indices.to(module.weight.device)
            module.weight.data = torch.index_select(module.weight.data, 0, pruned_channel_indices)
            print(module.weight.data.shape)
            if module.bias is not None:
                module.bias.data = torch.index_select(module.bias.data, 0, pruned_channel_indices)

            if module.running_mean is not None:
                module.running_mean.data = torch.index_select(module.running_mean.data, 0,
                                                                     pruned_channel_indices)
            if module.running_var is not None:
                module.running_var.data = torch.index_select(module.running_var.data, 0,
                                                                    pruned_channel_indices)
    return model



class GlobalConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(GlobalConvBlock, self).__init__()
        pad0 = int((kernel_size[0] - 1) / 2)
        pad1 = int((kernel_size[1] - 1) / 2)

        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                  padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                  padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                  padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                  padding=(pad0, 0))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.groups == m.in_channels:  # Depthwise convolution
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                else:  # Pointwise convolution
                    n = m.in_channels * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        # combine two paths
        x = x_l + x_r
        return x





class ResidualBlock(nn.Module):
    def __init__(self, indim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(indim, indim * 2, kernel_size=(1,1),bias=False)
        self.norm1 = nn.BatchNorm2d(indim * 2)
        self.relu1 = nn.LeakyReLU(0.2,inplace=True)

        self.conv2 = nn.Conv2d(indim * 2, indim * 2, kernel_size=(3,3), padding=1,bias=False)
        self.norm2 = nn.BatchNorm2d(indim * 2)
        self.relu2 = nn.LeakyReLU(0.2,inplace=True)

        self.conv3 = nn.Conv2d(indim * 2, indim, kernel_size=(1,1),bias=False)
        self.norm3 = nn.BatchNorm2d(indim)
        self.relu3 = nn.LeakyReLU(0.2,inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu1(residual)
        residual = self.conv2(residual)
        residual = self.relu2(residual)
        residual = self.conv3(residual)
        residual = self.relu3(residual)
        out = x + residual
        return out

class ResidualBlock_D(nn.Module):
    def __init__(self, indim):
        super(ResidualBlock_D, self).__init__()
        self.conv1 = nn.Conv2d(indim, indim * 2, kernel_size=(1,1),bias=False)
        self.norm1 = nn.BatchNorm2d(indim * 2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(indim * 2, indim * 2, kernel_size=(3,3), padding=1,bias=False)
        self.norm2 = nn.BatchNorm2d(indim * 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(indim * 2, indim, kernel_size=(1,1),bias=False)
        self.norm3 = nn.BatchNorm2d(indim)
        self.relu3 = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu1(residual)
        residual = self.conv2(residual)
        residual = self.relu2(residual)
        residual = self.conv3(residual)
        residual = self.relu3(residual)
        out = x + residual
        return out


import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)




class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)





class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
        # return x * y


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob=0.3, dilated=2):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                   dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                   dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        # if (self.dropout.p != 0):
        #     output = self.dropout(output)

        return F.relu(output + input)  # +input = identity (residual connection)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class ShuffleResidual(nn.Module):
    def __init__(self, inp):
        super(ShuffleResidual, self).__init__()
        oup=inp
        stride=1
        self.stride = stride
        assert stride in [1, 2]
        expand_ratio=2
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,  bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv1 = nn.Conv2d(inp, inp, kernel_size=(3, 3),padding=1 ,  bias=False)
            self.norm1 = nn.BatchNorm2d(inp)
            self.relu1 = nn.ReLU6()
            self.conv2 = nn.Conv2d(inp , inp*2, kernel_size=1, bias=False)
            self.norm2 = nn.BatchNorm2d(inp*2)
            self.relu2 = nn.ReLU6()
            self.conv3 = nn.Conv2d(inp*2, inp, kernel_size=1, bias=False)
            self.norm3 = nn.BatchNorm2d(inp)
            self.relu3 = nn.ReLU6()
            # self.se_block = eca_layer(inp)
            self.conv4 = nn.Conv2d(inp,inp, kernel_size=(3, 3),padding=1, bias=False)
            self.norm4 = nn.BatchNorm2d(inp)
            self.relu4 = nn.ReLU6()


            # parameter initialization
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.normal_(1.0, 0.02)
                    m.bias.data.fill_(0)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu1(residual)
        # residual = self.quant(residual)
        residual = self.conv2(residual)
        residual = self.relu2(residual)
        # residual = self.quant(residual)
        residual = self.conv3(residual)
        residual = self.relu3(residual)

        # residual = self.se_block(residual)
        residual = self.conv4(residual)
        residual = self.relu4(residual)

        # residual = self.quant(residual)
        out = x + residual
        return out


class NetS(nn.Module):
    def __init__(self,ndf):
        super(NetS, self).__init__()
        conv_class = nn.Conv2d

        ResidualBlock=ShuffleResidual
        ResidualBlock_D = ShuffleResidual
        self.convblock1 = nn.Sequential(
            # input is (channel_dim) x 128 x 128
            nn.Conv2d(channel_dim, ndf, 7, 2, 3, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
        )

        self.convblock1_1 = ResidualBlock(ndf)
        self.convblock2 = nn.Sequential(
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, (5,5), 2, 2, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
        )
        self.convblock2_1 = ResidualBlock(ndf*2)
        self.convblock3 = nn.Sequential(
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, (5,5), 2, 2, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
        )
        self.convblock3_1 = ResidualBlock(ndf*4)
        self.convblock4 = nn.Sequential(
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, (5,5), 2, 2, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
        )
        self.convblock4_1 = ResidualBlock(ndf*8)
        self.convblock5 = nn.Sequential(
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 8,(5,5), 2, 2, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.SiLU(),
            # state size. (ndf*8) x 4 x 4
        )
        self.convblock5_1 =ResidualBlock(ndf*8)
        self.convblock6 = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 16, (4,4), 2, 1, bias=False),
            nn.BatchNorm2d(ndf*16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 2 x 2
        )
        self.convblock6_1 = nn.Sequential(
            # state size. (ndf*16) x 2 x 2
            nn.Conv2d(ndf * 16, ndf * 16, kernel_size=(1,1), bias=False),
            nn.BatchNorm2d(ndf*16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 2 x 2
        )
        self.convblock7 = nn.Sequential(
            # state size. (ndf*16) x 2 x 2
            nn.Conv2d(ndf * 16, ndf * 32, (3,3), 2, 1, bias=False),
            nn.BatchNorm2d(ndf*32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*32) x 1 x 1
        )
        # self.convblock7_1 = ResidualBlock(ndf*32)
        self.convblock8 = nn.Sequential(
            # state size. (ndf*32) x 1 x 1
            nn.Conv2d(ndf * 32, ndf * 8, kernel_size=(1,1), bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 1 x 1
        )


        self.deconvblock1 = nn.Sequential(
            # state size. (ngf*8) x 1 x 1
            nn.Conv2d(ndf * 8, ndf * 32, kernel_size=(1,1), bias=False),
            nn.BatchNorm2d(ndf*32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*32) x 1 x 1
        )
        self.deconvblock2 = nn.Sequential(
            # state size. (cat: ngf*32) x 1 x 1
            nn.Conv2d(ndf * 64 , ndf * 16, (3,3), 1, 1, bias=False),
            nn.BatchNorm2d(ndf*16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*16) x 2 x 2
        )
        self.deconvblock2_1 = nn.Sequential(
            # state size. (ndf*16) x 2 x 2
            nn.Conv2d(ndf * 16, ndf * 16, kernel_size=(1,1), bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 2 x 2
        )
        self.deconvblock3 = nn.Sequential(
            # state size. (cat: ngf*16) x 2 x 2
            nn.Conv2d(ndf * 16 * 2, ndf * 8, (3,3), 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.SiLU(),
            # state size. (ngf*8) x 4 x 4
        )
        self.deconvblock3_1 = ResidualBlock_D(ndf*8)
        self.deconvblock4 = nn.Sequential(
            # state size. (ngf*8) x 4 x 4
            GlobalConvBlock(ndf*8*2, ndf*8, (7, 7)),
            # nn.ConvTranspose2d(ndf * 8 * 2, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
        )
        self.deconvblock4_1 = ResidualBlock_D(ndf*8)
        self.deconvblock5 = nn.Sequential(
            # state size. (ngf*8) x 8 x 8
            GlobalConvBlock(ndf*8*2, ndf*4, (7, 7)),
            # nn.ConvTranspose2d(ndf * 8 * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
        )
        self.deconvblock5_1 = ResidualBlock_D(ndf*4)
        self.deconvblock6 = nn.Sequential(
            # state size. (ngf*4) x 16 x 16
            GlobalConvBlock(ndf*4*2, ndf*2, (9, 9)),
            # nn.ConvTranspose2d(ndf * 4 * 2, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
        )
        self.deconvblock6_1 = ResidualBlock_D(ndf*2)
        self.deconvblock7 = nn.Sequential(
            # state size. (ngf*2) x 32 x 32
            GlobalConvBlock(ndf*2*2, ndf, (9, 9)),
            # nn.ConvTranspose2d(ndf * 2 * 2,     ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
        )
        self.deconvblock7_1 = ResidualBlock_D(ndf)
        self.deconvblock8 = nn.Sequential(
            # state size. (ngf) x 64 x 64
            GlobalConvBlock(ndf*2, ndf, (11, 11)),
            # nn.ConvTranspose2d( ndf * 2, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            # state size. (ngf) x 128 x 128
        )
        self.deconvblock8_1 = ResidualBlock_D(ndf)
        self.deconvblock9 = nn.Sequential(
            # state size. (ngf) x 128 x 128
            nn.Conv2d(ndf, classes, (5,5), 1, 2, bias=False),
            # state size. (channel_dim) x 128 x 128
            # nn.Sigmoid()
        )
        # self.quant = torch.quantization.QuantStub()
        # # DeQuantStub converts tensors from quantized to floating point
        # self.dequant = torch.quantization.DeQuantStub()
        for m in self.modules():
            if isinstance(m, conv_class):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, input):
        # for now it only supports one GPU
        encoder1 = self.convblock1(input)
        encoder1 = self.convblock1_1(encoder1)
        encoder2 = self.convblock2(encoder1)
        encoder2 = self.convblock2_1(encoder2)
        encoder3 = self.convblock3(encoder2)
        encoder3 = self.convblock3_1(encoder3)
        encoder4 = self.convblock4(encoder3)
        encoder4 = self.convblock4_1(encoder4)
        encoder5 = self.convblock5(encoder4)
        encoder5 = self.convblock5_1(encoder5)
        encoder6 = self.convblock6(encoder5)
        encoder6 = self.convblock6_1(encoder6) + encoder6
        encoder7 = self.convblock7(encoder6)
        encoder8 = self.convblock8(encoder7)

        decoder1 = self.deconvblock1(encoder8)
        decoder1 = torch.cat([encoder7, decoder1], 1)
        decoder1 = F.upsample(decoder1, size=encoder6.size()[2:], mode='bilinear')
        decoder2 = self.deconvblock2(decoder1)
        decoder2 = self.deconvblock2_1(decoder2) + decoder2
        # concatenate along depth dimension
        decoder2 = torch.cat([encoder6, decoder2], 1)
        decoder2 = F.upsample(decoder2, size=encoder5.size()[2:], mode='bilinear')
        decoder3 = self.deconvblock3(decoder2)
        decoder3 = self.deconvblock3_1(decoder3)
        decoder3 = torch.cat([encoder5, decoder3], 1)
        decoder3 = F.upsample(decoder3, size=encoder4.size()[2:], mode='bilinear')
        decoder4 = self.deconvblock4(decoder3)
        decoder4 = self.deconvblock4_1(decoder4)
        decoder4 = torch.cat([encoder4, decoder4], 1)
        decoder4 = F.upsample(decoder4, size=encoder3.size()[2:], mode='bilinear')
        decoder5 = self.deconvblock5(decoder4)
        decoder5 = self.deconvblock5_1(decoder5)
        decoder5 = torch.cat([encoder3, decoder5], 1)
        decoder5 = F.upsample(decoder5, size=encoder2.size()[2:], mode='bilinear')
        decoder6 = self.deconvblock6(decoder5)
        decoder6 = self.deconvblock6_1(decoder6)
        decoder6 = torch.cat([encoder2, decoder6], 1)
        decoder6 = F.upsample(decoder6, size=encoder1.size()[2:], mode='bilinear')
        decoder7 = self.deconvblock7(decoder6)
        decoder7 = self.deconvblock7_1(decoder7)
        decoder7 = torch.cat([encoder1, decoder7], 1)
        decoder7 = F.upsample(decoder7, size=input.size()[2:], mode='bilinear')
        decoder8 = self.deconvblock8(decoder7)
        decoder8 = self.deconvblock8_1(decoder8)
        decoder9 = self.deconvblock9(decoder8)

        return decoder9

class NetS_DNA(nn.Module):
    def __init__(self,bn=False):
        super(NetS_DNA, self).__init__()

        BN=nn.BatchNorm2d

        self.convblock1 = nn.Sequential(
            # input is (channel_dim) x 128 x 128
            nn.Conv2d(channel_dim, ndf, 7, 2, 3, bias=False),
            #IBN(ndf, 0.4),
            BN(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
        )
        self.convblock1_1 = ResidualBlock(ndf)
        self.convblock2 = nn.Sequential(
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 5, 2, 2, bias=False),
            BN(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
        )
        self.convblock2_1 = ResidualBlock(ndf*2)
        self.convblock3 = nn.Sequential(
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 5, 2, 2, bias=False),
            #IBN(ndf*4, 0.4),
            BN(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
        )
        self.convblock3_1 = ResidualBlock(ndf*4)
        self.convblock4 = nn.Sequential(
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
        )
        self.convblock4_1 = ResidualBlock(ndf*8)
        self.convblock5 = nn.Sequential(
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 8, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
        )
        self.convblock5_1 = ResidualBlock(ndf*8)
        self.convblock6 = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 2 x 2
        )
        self.convblock6_1 = nn.Sequential(
            # state size. (ndf*16) x 2 x 2
            nn.Conv2d(ndf * 16, ndf * 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(ndf*16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 2 x 2
        )
        self.convblock7 = nn.Sequential(
            # state size. (ndf*16) x 2 x 2
            nn.Conv2d(ndf * 16, ndf * 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*32) x 1 x 1
        )
        # self.convblock7_1 = ResidualBlock(ndf*32)
        self.convblock8 = nn.Sequential(
            # state size. (ndf*32) x 1 x 1
            nn.Conv2d(ndf * 32, ndf * 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 1 x 1
        )


        self.deconvblock1 = nn.Sequential(
            # state size. (ngf*8) x 1 x 1
            nn.Conv2d(ndf * 8, ndf * 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(ndf*32),
            nn.ReLU(True),
            # state size. (ngf*32) x 1 x 1
        )
        self.deconvblock2 = nn.Sequential(
            # state size. (cat: ngf*32) x 1 x 1
            nn.Conv2d(ndf * 64 , ndf * 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf*16),
            nn.ReLU(True),
            # state size. (ngf*16) x 2 x 2
        )
        self.deconvblock2_1 = nn.Sequential(
            # state size. (ndf*16) x 2 x 2
            nn.Conv2d(ndf * 16, ndf * 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.ReLU(inplace=True),
            # state size. (ndf*16) x 2 x 2
        )
        self.deconvblock3 = nn.Sequential(
            # state size. (cat: ngf*16) x 2 x 2
            nn.Conv2d(ndf * 16 * 2, ndf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
        )
        self.deconvblock3_1 = ResidualBlock_D(ndf*8)
        self.deconvblock4 = nn.Sequential(
            # state size. (ngf*8) x 4 x 4
            GlobalConvBlock(ndf*8*2, ndf*8, (7, 7)),
            # nn.ConvTranspose2d(ndf * 8 * 2, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
        )
        self.deconvblock4_1 = ResidualBlock_D(ndf*8)
        self.deconvblock5 = nn.Sequential(
            # state size. (ngf*8) x 8 x 8
            GlobalConvBlock(ndf*8*2, ndf*4, (7, 7)),
            # nn.ConvTranspose2d(ndf * 8 * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
        )
        self.deconvblock5_1 = ResidualBlock_D(ndf*4)
        self.deconvblock6 = nn.Sequential(
            # state size. (ngf*4) x 16 x 16
            GlobalConvBlock(ndf*4*2, ndf*2, (9, 9)),
            # nn.ConvTranspose2d(ndf * 4 * 2, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
        )
        self.deconvblock6_1 = ResidualBlock_D(ndf*2)
        self.deconvblock7 = nn.Sequential(
            # state size. (ngf*2) x 32 x 32
            GlobalConvBlock(ndf*2*2, ndf, (9, 9)),
            # nn.ConvTranspose2d(ndf * 2 * 2,     ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
        )
        self.deconvblock7_1 = ResidualBlock_D(ndf)
        self.deconvblock8 = nn.Sequential(
            # state size. (ngf) x 64 x 64
            GlobalConvBlock(ndf*2, ndf, (11, 11)),
            # nn.ConvTranspose2d( ndf * 2, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            # state size. (ngf) x 128 x 128
        )
        self.deconvblock8_1 = ResidualBlock_D(ndf)
        self.deconvblock9 = nn.Sequential(
            # state size. (ngf) x 128 x 128
            nn.Conv2d( ndf, classes, 5, 1, 2, bias=False),
            # state size. (channel_dim) x 128 x 128
            # nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            #elif isinstance(m, nn.BatchNorm2d):
            #    m.weight.data.normal_(1.0, 0.02)
            #    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, input):
        # for now it only supports one GPU
        encoder1 = self.convblock1(input)
        encoder1 = self.convblock1_1(encoder1)
        encoder2 = self.convblock2(encoder1)
        encoder2 = self.convblock2_1(encoder2)
        encoder3 = self.convblock3(encoder2)
        encoder3 = self.convblock3_1(encoder3)
        encoder4 = self.convblock4(encoder3)
        encoder4 = self.convblock4_1(encoder4)
        encoder5 = self.convblock5(encoder4)
        encoder5 = self.convblock5_1(encoder5)
        encoder6 = self.convblock6(encoder5)
        encoder6 = self.convblock6_1(encoder6) + encoder6
        encoder7 = self.convblock7(encoder6)
        encoder8 = self.convblock8(encoder7)

        decoder1 = self.deconvblock1(encoder8)
        decoder1 = torch.cat([encoder7, decoder1], 1)
        decoder1 = F.upsample(decoder1, size=encoder6.size()[2:], mode='bilinear')
        decoder2 = self.deconvblock2(decoder1)
        decoder2 = self.deconvblock2_1(decoder2) + decoder2
        # concatenate along depth dimension
        decoder2 = torch.cat([encoder6, decoder2], 1)
        decoder2 = F.upsample(decoder2, size=encoder5.size()[2:], mode='bilinear')
        decoder3 = self.deconvblock3(decoder2)
        decoder3 = self.deconvblock3_1(decoder3)
        decoder3 = torch.cat([encoder5, decoder3], 1)
        decoder3 = F.upsample(decoder3, size=encoder4.size()[2:], mode='bilinear')
        decoder4 = self.deconvblock4(decoder3)
        decoder4 = self.deconvblock4_1(decoder4)
        decoder4 = torch.cat([encoder4, decoder4], 1)
        decoder4 = F.upsample(decoder4, size=encoder3.size()[2:], mode='bilinear')
        decoder5 = self.deconvblock5(decoder4)
        decoder5 = self.deconvblock5_1(decoder5)
        decoder5 = torch.cat([encoder3, decoder5], 1)
        decoder5 = F.upsample(decoder5, size=encoder2.size()[2:], mode='bilinear')
        decoder6 = self.deconvblock6(decoder5)
        decoder6 = self.deconvblock6_1(decoder6)
        decoder6 = torch.cat([encoder2, decoder6], 1)
        decoder6 = F.upsample(decoder6, size=encoder1.size()[2:], mode='bilinear')
        decoder7 = self.deconvblock7(decoder6)
        decoder7 = self.deconvblock7_1(decoder7)
        decoder7 = torch.cat([encoder1, decoder7], 1)
        decoder7 = F.upsample(decoder7, size=input.size()[2:], mode='bilinear')
        decoder8 = self.deconvblock8(decoder7)
        decoder8 = self.deconvblock8_1(decoder8)
        decoder9 = self.deconvblock9(decoder8)

        return decoder9,decoder4,decoder8


class NetC(nn.Module):
    def __init__(self, ndf):
        super(NetC, self).__init__()

        self.convblock1 = nn.Sequential(
            # input is (channel_dim) x 128 x 128
            nn.Conv2d(classes, ndf, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf) x 64 x 64
        )

        self.convblock2 = nn.Sequential(
            # state size. (ndf * 2) x 64 x 64
            nn.Conv2d(ndf * 1, ndf * 2, (5,5), 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*2) x 32 x 32
        )

        self.convblock3 = nn.Sequential(
            # state size. (ndf * 4) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, (4,4), 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*4) x 16 x 16
        )



        self.convblock4 = nn.Sequential(
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, (4,4), 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*8) x 8 x 8
        )

        self.convblock5 = nn.Sequential(
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 8, (4,4), 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*16) x 4 x 4
        )

        self.convblock6 = nn.Sequential(
            # state size. (ndf*32) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 8, (3,3), 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*32) x 2 x 2
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.groups == m.in_channels:  # Depthwise convolution
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                else:  # Pointwise convolution
                    n = m.in_channels * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()


    def forward(self, input):
        batchsize = input.size()[0]
        out1 = self.convblock1(input)

        # out1 = self.convblock1_1(out1)
        out2 = self.convblock2(out1)

        # out2 = self.convblock2_1(out2)
        out3 = self.convblock3(out2)

        # out3 = self.convblock3_1(out3)
        out4 = self.convblock4(out3)

        # out4 = self.convblock4_1(out4)
        out5 = self.convblock5(out4)

        # out5 = self.convblock5_1(out5)
        out6 = self.convblock6(out5)  # [4,256,4,4]

        # out6 = self.convblock6_1(out6) + out6
        output = torch.cat((input.view(batchsize, -1), 1 * out1.view(batchsize, -1),
                            2 * out2.view(batchsize, -1), 2 * out3.view(batchsize, -1),
                            2 * out4.view(batchsize, -1), 2 * out5.view(batchsize, -1),
                            4 * out6.view(batchsize, -1)), 1)

        # print(c.shape)

        return output



