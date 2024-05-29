import random, time, os, datetime, sys

from torch.autograd import Variable

import torch.nn as nn
from torch.nn.functional import normalize


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.values = []
        self.counter = 0

    def append(self, val):
        self.values.append(val)
        self.counter += 1

    @property
    def val(self):
        return self.values[-1]

    @property
    def avg(self):
        return sum(self.values) / len(self.values)

    @property
    def last_avg(self):
        if self.counter == 0:
            return self.latest_avg
        else:
            self.latest_avg = sum(self.values[-self.counter:]) / self.counter
            self.counter = 0
            return self.latest_avg


def save_ckpt(epoch, netG, netD,
              optimizer_G, optimizer_D, optimizer_gamma,
              lr_scheduler_G, lr_scheduler_D, lr_scheduler_gamma,
              loss_G_lst, loss_G_perceptual_lst, loss_G_GAN_lst, loss_D_lst, channel_number_lst,
              path):
    ckpt = {
        'epoch': epoch,
        'netG': netG.state_dict(),
        'netD': netD.state_dict(),
        'optimizer_G': optimizer_G.state_dict(),
        'optimizer_D': optimizer_D.state_dict(),
        'optimizer_gamma': optimizer_gamma.state_dict(),
        'lr_scheduler_G': lr_scheduler_G.state_dict(),
        'lr_scheduler_D': lr_scheduler_D.state_dict(),
        'lr_scheduler_gamma': lr_scheduler_gamma.state_dict(),
        'loss_G_lst': loss_G_lst,
        'loss_G_perceptual_lst': loss_G_perceptual_lst,
        'loss_G_GAN_lst': loss_G_GAN_lst,
        'loss_D_lst': loss_D_lst,
        'channel_number_lst': channel_number_lst,
    }
    torch.save(ckpt, path)


def load_ckpt(netG, netD,
              optimizer_G, optimizer_D, optimizer_gamma,
              lr_scheduler_G, lr_scheduler_D, lr_scheduler_gamma, path):
    if not os.path.isfile(path):
        raise Exception('No such file: %s' % path)
    print("===>>> loading checkpoint from %s" % path)
    ckpt = torch.load(path)

    epoch = ckpt['epoch']
    loss_G_lst = ckpt['loss_G_lst']
    loss_G_perceptual_lst = ckpt['loss_G_perceptual_lst']
    loss_G_GAN_lst = ckpt['loss_G_GAN_lst']
    loss_D_lst = ckpt['loss_D_lst']
    channel_number_lst = ckpt['channel_number_lst']
    # best_FID = ckpt['best_FID']

    netG.load_state_dict(ckpt['netG'])
    netD.load_state_dict(ckpt['netD'])
    optimizer_G.load_state_dict(ckpt['optimizer_G'])
    optimizer_D.load_state_dict(ckpt['optimizer_D'])
    optimizer_gamma.load_state_dict(ckpt['optimizer_gamma'])
    lr_scheduler_G.load_state_dict(ckpt['lr_scheduler_G'])
    lr_scheduler_D.load_state_dict(ckpt['lr_scheduler_D'])
    lr_scheduler_gamma.load_state_dict(ckpt['lr_scheduler_gamma'])

    return epoch, loss_G_lst, loss_G_perceptual_lst, loss_G_GAN_lst, loss_D_lst, channel_number_lst


def save_ckpt_finetune(epoch, netG, netD,
                       optimizer_G, optimizer_D,
                       lr_scheduler_G, lr_scheduler_D,
                       loss_G_lst, loss_G_perceptual_lst, loss_G_GAN_lst, loss_D_lst,
                       best_FID,
                       path):
    ckpt = {
        'epoch': epoch,
        'netG': netG.state_dict(),
        'netD': netD.state_dict(),
        'optimizer_G': optimizer_G.state_dict(),
        'optimizer_D': optimizer_D.state_dict(),
        'lr_scheduler_G': lr_scheduler_G.state_dict(),
        'lr_scheduler_D': lr_scheduler_D.state_dict(),
        'loss_G_lst': loss_G_lst,
        'loss_G_perceptual_lst': loss_G_perceptual_lst,
        'loss_G_GAN_lst': loss_G_GAN_lst,
        'loss_D_lst': loss_D_lst,
        'best_FID': best_FID,
    }
    torch.save(ckpt, path)


def load_ckpt_finetune(netG, netD,
                       optimizer_G, optimizer_D,
                       lr_scheduler_G, lr_scheduler_D,
                       path):
    if not os.path.isfile(path):
        raise Exception('No such file: %s' % path)
    print("===>>> loading checkpoint from %s" % path)
    ckpt = torch.load(path)
    epoch = ckpt['epoch']
    loss_G_lst = ckpt['loss_G_lst']
    loss_G_perceptual_lst = ckpt['loss_G_perceptual_lst']
    loss_G_GAN_lst = ckpt['loss_G_GAN_lst']
    loss_D_lst = ckpt['loss_D_lst']

    best_FID = ckpt['best_FID']

    netG.load_state_dict(ckpt['netG'])
    netD.load_state_dict(ckpt['netD'])

    optimizer_G.load_state_dict(ckpt['optimizer_G'])
    optimizer_D.load_state_dict(ckpt['optimizer_D'])

    lr_scheduler_G.load_state_dict(ckpt['lr_scheduler_G'])
    lr_scheduler_D.load_state_dict(ckpt['lr_scheduler_D'])
    return epoch, loss_G_lst, loss_G_perceptual_lst, loss_G_GAN_lst, loss_D_lst, best_FID


def tensor2image(tensor):
    image = 127.5 * (tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8)


class ReplayBuffer():
    '''
    follow Shrivastava et al.’s strategy:
    update D using a history of generated images, rather than the ones produced by the latest generators.
    '''

    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # torch.nn.init.normal(m.weight.data, 0.0, 0.02)
        torch.nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


def soft_threshold(w, th):
    '''
    pytorch soft-sign function
    '''
    with torch.no_grad():
        temp = torch.abs(w) - th
        # print('th:', th)
        # print('temp:', temp.size())
        return torch.sign(w) * nn.functional.relu(temp)


count_ops = 0
num_ids = 0


def get_feature_hook(self, _input, _output):
    global count_ops, num_ids
    # print('------>>>>>>')
    # print('{}th node, input shape: {}, output shape: {}, input channel: {}, output channel {}'.format(
    # 	num_ids, _input[0].size(2), _output.size(2), _input[0].size(1), _output.size(1)))
    # print(self)
    delta_ops = self.in_channels * self.out_channels * self.kernel_size[0] * self.kernel_size[1] * _output.size(
        2) * _output.size(3) / self.groups
    count_ops += delta_ops
    # print('ops is {:.6f}M'.format(delta_ops / 1024.  /1024.))
    num_ids += 1


# print('')

def measure_model(net, H_in, W_in):
    import torch
    import torch.nn as nn
    _input = torch.randn((1, 1, H_in, W_in))
    # _input, net = _input.cpu(), net.cpu()
    hooks = []
    for module in net.named_modules():
        if isinstance(module[1], nn.Conv2d) or isinstance(module[1], nn.ConvTranspose2d):
            # print(module)
            hooks.append(module[1].register_forward_hook(get_feature_hook))

    _out = net(_input)
    global count_ops
    print('count_ops: {:.6f}G'.format(count_ops / 1024. / 1024./1024.))  # in Million
    return count_ops

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print("模型总大小为：{:.3f}MB".format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)


def show_sparsity(model, save_name, model_path=None):
    # load model if necessary:
    if model_path is not None:
        if not os.path.exists(model_path):
            raise Exception("G model path doesn't exist at %s!" % model_path)
        print('Loading generator from %s' % model_path)
        model.load_state_dict(torch.load(model_path))

    # get all scaler parameters form the network:
    scaler_list = []
    for m in model.modules():
        if isinstance(m, torch.nn.InstanceNorm2d) and m.weight is not None:
            m_cpu = m.weight.data.cpu().numpy().squeeze()
            # print('m_cpu:', type(m_cpu), m_cpu.shape)
            scaler_list.append(m_cpu)
    all_scaler = np.concatenate(scaler_list, axis=0)
    print('all_scaler:', all_scaler.shape, 'L0 (sum):', np.sum(all_scaler != 0), 'L1 (mean):',
          np.mean(np.abs(all_scaler)))

    # save npy and plt png:
    # np.save(save_name + '.npy', all_scaler)
    n, bins, patches = plt.hist(all_scaler, 50)
    # print(n)
    plt.savefig(save_name + '.png')
    plt.close()

    return all_scaler


def none_zero_channel_num(model, model_path=None):
    # load model if necessary:
    if model_path is not None:
        if not os.path.exists(model_path):
            raise Exception("G model path doesn't exist at %s!" % model_path)
        print('Loading generator from %s' % model_path)
        model.load_state_dict(torch.load(model_path))

    # get all scaler parameters form the network:
    scaler_list = []
    for m in model.modules():
        if isinstance(m, torch.nn.InstanceNorm2d) and m.weight is not None:
            m_cpu = m.weight.data.cpu().numpy().squeeze()
            # print('m_cpu:', type(m_cpu), m_cpu.shape)
            scaler_list.append(m_cpu)
    all_scaler = np.concatenate(scaler_list, axis=0)
    l0norm = np.sum(all_scaler != 0)
    print('all_scaler:', all_scaler.shape, 'L0 (sum):', l0norm, 'L1 (mean):', np.mean(np.abs(all_scaler)))

    return l0norm


def create_dir(_path):
    if not os.path.exists(_path):
        os.makedirs(_path)


def fourD2threeD(batch, n_row=10):
    '''
    Convert a batch of images (N,W,H,C) to a single big image (W*n, H*m, C)
    Input:
        batch: type=ndarray, shape=(N,W,H,C)
    Return:
        rows: type=ndarray, shape=(W*n, H*m, C)
    '''
    N = batch.shape[0]
    img_list = np.split(batch, N)
    for i, img in enumerate(img_list):
        img_list[i] = img.squeeze(axis=0)
    one_row = np.concatenate(img_list, axis=1)
    # print('one_row:', one_row.shape)
    row_list = np.split(one_row, n_row, axis=1)
    rows = np.concatenate(row_list, axis=0)
    return rows


def layer_param_num(model, param_name=['weight']):
    count_res = {}
    for name, W in model.named_parameters():
        if name.strip().split(".")[-1] in param_name and name.strip().split(".")[-2][:2] != "bn" and W.dim() > 1:
            # W_nz = torch.nonzero(W.data)
            W_nz = torch.flatten(W.data)
            if W_nz.dim() > 0:
                count_res[name] = W_nz.shape[0]
    return count_res


def model_param_num(model, param_name=['weight']):
    '''
    Find parameter numbers in the model. in Million.
    '''
    layer_size_dict = layer_param_num(model, param_name)
    return sum(layer_size_dict.values()) / 1024 / 1024

import os
import pathlib

import seaborn as sns

import cv2
import nibabel as nib
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标签，如果想要用新罗马字体，改成 Times New Roman
#plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
#plt.rcParams['axes.unicode_minus']=False #解决负号‘-’显示为方块的问题
import torch
from pathlib import Path

from tqdm import tqdm


from sklearn.model_selection import train_test_split
import torch.nn.functional as F


import imgaug
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def mergeChannels(array, size):
    c0 = array[:,0,:,:].reshape(-1, 1, size, size)
    c1 = array[:,1,:,:].reshape(-1, 1, size, size)

    c0[c0>=0.5] = 1
    c0[c0<0.5] = 0

    c1[c1>=0.5] = 2
    c1[c1<0.5] = 0

    array = np.hstack((c0, c1))

    array = np.amax(array, axis=1)

    # c0 = c0.flatten()
    # c1 = c1.flatten()
    # array = array.flatten()

    # for i in range(array.shape[0]):
    #     if (array[i] == 0):
    #         array[i] = c0[i]
    #     else:
    #         array[i] = c1[i]

    return array.reshape(-1, 1, size, size)

def visualizeMasks(epoch,fed_round,client,img,gt,target, output, size,flag):
    img = img.numpy()
    gt = gt.numpy()
    output = output.numpy()
    target=target.numpy()
    #print(img.shape,gt.shape)
    for i in range(img.shape[0]):

        image=img[i].reshape(-1, 1, size, size)
        mask=gt[i].reshape(-1, 1, size, size)

        # 真实
        trLr = target[i].reshape(-1, 2, size, size)
        tr1Lr, tr2Lr = trLr[:, 0, :, :], trLr[:, 1, :, :]
        #真实合并
        trRr=mergeChannels(trLr, size)


        #预测
        imgLr=output[i].reshape(-1, 2, size, size)  #[1,2,256,256]
        imgM1r, imgM2r = imgLr[:, 0, :, :], imgLr[:, 1, :, :]

        #预测合并
        imgRr = mergeChannels(imgLr, size)



        # print(imgM1r.shape)
        # print(imgM2r.shape)



        # print(imgRr.shape)
        f = plt.figure()
        # f.suptitle('GT:{} LC:{} TC:{} CC:{}'.format([np.min(gt), np.max(gt)], [np.min(imgM1r), np.max(imgM1r)],
        #                                                 [np.min(imgM2r), np.max(imgM2r)], [np.min(imgRr), np.max(imgRr)]), fontsize=20)

        f.add_subplot(2, 3, 1)
        plt.title('gt')
        plt.imshow(mask.reshape(size, size))

        f.add_subplot(2, 3, 2)
        plt.title('gt-infect-1')
        plt.imshow(tr1Lr.reshape(size, size))

        f.add_subplot(2, 3, 3)
        plt.title('gt-infect-2')
        plt.imshow(tr2Lr.reshape(size, size))


        f.add_subplot(2, 3, 4)
        plt.title('pred')
        plt.imshow(imgRr.reshape(size, size))
        f.add_subplot(2, 3, 5)
        plt.title('pred-infect-1')
        plt.imshow(imgM1r.reshape(size, size))
        f.add_subplot(2, 3, 6)
        plt.title('pred-infect-2')
        plt.imshow(imgM2r.reshape(size, size))
        plt.show(block=False)

        if (flag == 1):
            plt.savefig('./result/fed_round-{}-client-{}-epoch-{}-{}-gt-pred.png'.format(fed_round,client,epoch,i))
        else:
            plt.savefig('./result/fed_round-{}-client-{}-{}-val-gt-pred.png'.format(fed_round,client,i))


def read_nii(filepath):
    '''
    Reads .nii file and returns pixel array
    '''
    ct_scan = nib.load(filepath)
    # print(ct_scan.shape)
    array = ct_scan.get_fdata()
    # array= np.rot90(np.array(array))

    return (array)


def change_img_to_label_path(path):
    """
    Replaces imagesTr with labelsTr
    """
    parts = list(path.parts)  # get all directories whithin the path
    parts[parts.index("ct_scan")] = "infection_mask"  # Replace imagesTr with labelsTr
    return pathlib.Path(*parts)  # Combine list back into a Path object

def change_img_to_label_path1(path):
    parts = list(path.parts)
    parts[parts.index("data")] = "masks"
    return Path(*parts)

def crete_split(path, path1, path2, save_root):
    if (not os.path.isdir(save_root)):
        print('Extracting images from volume, on disk.')
        print()
        for counter, path_to_ct_data in enumerate(path):
            if (path == path1):
                label2 = pathlib.Path("../covid-19_data/covid19-40/infection_mask/")
                sample_path_label2 = list(label2.glob("tr*"))[0]
                path_to_label = sample_path_label2
            else:
                path_to_label = change_img_to_label_path(path_to_ct_data)
            ct = read_nii(path_to_ct_data)  # type(ct) 是numpy类型
            infect = read_nii(path_to_label)

            ct_data = ct
            mask_data = infect

            if counter < len(path) * 0.8:
                current_path = save_root / "train" / str(counter)
            else:
                current_path = save_root / "val" / str(counter)
            for i in range(ct_data.shape[-1]):
                slice = ct_data[:, :, i]
                mask = mask_data[:, :, i]
                slice_path = current_path / "data"
                mask_path = current_path / "masks"
                slice_path.mkdir(parents=True, exist_ok=True)
                mask_path.mkdir(parents=True, exist_ok=True)

                np.save(slice_path / str(i), slice)
                np.save(mask_path / str(i), mask)
    else:
        print('Extracted images already present on the disk.')

#

#客户端数据处理
def crete_split1(path, save_root):
    if (not os.path.isdir(save_root)):
        print('Extracting images from volume, on disk.')
        print()
        for counter, path_to_ct_data in enumerate(path):

            ct = read_nii(path_to_ct_data)  # type(ct) 是numpy类型
            ct_data = ct

            if counter < len(path) * 0.8:
                current_path = save_root / "train" / str(counter)
            else:
                current_path = save_root / "val" / str(counter)
            for i in range(ct_data.shape[-1]):
                slice = ct_data[:, :, i]
                slice_path = current_path / "data"
                slice_path.mkdir(parents=True, exist_ok=True)
                np.save(slice_path / str(i), slice)
    else:
        print('Extracted images already present on the disk.')


#给数据集1划分子集
def crete_client(path,size):
    lungs = []
    infections = []

    all_files = extract_files(path)

    for idx in range(len(all_files)):
        file_path = all_files[idx]
        mask_path = change_img_to_label_path1(file_path)

        slice = np.load(file_path)
        gt = np.load(mask_path)
        img = normalize(slice)  # 标准化和去除肺部阴影操作
        img = cv2.resize(img, (size, size))
        gt = cv2.resize(gt, (size, size), interpolation=cv2.INTER_NEAREST)

        lungs.append(img)
        infections.append(gt)

    return np.array(lungs),np.array(infections)

def extract_files(root):
    files = []
    for subject in root.glob("*"):  # Iterate over the subjects
        slice_path = subject / "data"  # Get the slices for current subject
        for slice in slice_path.glob("*"):
            files.append(slice)
    return files

#给数据集1划分子集
def crete_client1(path,size):
    lungs=[]
    infections=[]
    all_files = extract_files(path)
    for idx in range(len(all_files)):
        file_path = all_files[idx]
        mask_path = change_img_to_label_path1(file_path)

        slice = np.load(file_path)
        gt = np.load(mask_path)
        img = normalize(slice)  # 标准化和去除肺部阴影操作
        img = cv2.resize(img, (size, size))
        gt = cv2.resize(gt, (size, size), interpolation=cv2.INTER_NEAREST)

        lungs.append(img)
        infections.append(gt)
            #targets.append(target)
    lungs = np.array(lungs)
    infections = np.array(infections)

    #分法大小1数量不一致不想交
    train_data1, train_data2, train_label1, train_label2 = train_test_split(lungs, infections, test_size=0.4,
                                                                        random_state=42)

    train_data1, train_data3, train_label1, train_label3 = train_test_split(train_data1, train_label1, test_size=0.4,
                                                                            random_state=42)

    #train_data1, train_data4, train_label1, train_label4 = train_test_split(train_data1, train_label1, test_size=0.2,random_state=42)

    #train_data1, train_data5, train_label1, train_label5 = train_test_split(train_data1, train_label1, test_size=0.2,random_state=42)
    #data1_client=[(train_data1,train_label1),(train_data2,train_label2),(train_data3,train_label3)]

    labeled_client1_data,unlabeled_client1_data,labeled_client1_label,unlabeled_client1_label=train_test_split(train_data1, train_label1, test_size=0.7,random_state=42)
    labeled_client2_data,unlabeled_client2_data, labeled_client2_label,  unlabeled_client2_label = train_test_split(train_data2, train_label2, test_size=0.7,random_state=42)
    labeled_client3_data,unlabeled_client3_data,  labeled_client3_label, unlabeled_client3_label = train_test_split(train_data3, train_label3, test_size=0.7, random_state=42)

    data_client=[(labeled_client1_data,labeled_client1_label),(unlabeled_client1_data,unlabeled_client1_label),
                 (labeled_client2_data, labeled_client2_label),(unlabeled_client2_data, unlabeled_client2_label),
                 (labeled_client3_data, labeled_client3_label),(unlabeled_client3_data, unlabeled_client3_label)]
    #分法2大小数量一致不想交

    return data_client


def crete_two(dataset):
    lungs=[]
    infections=[]
    for i in range(0, len(dataset)):
        img, target, gt,_,_ = dataset[i]
        lungs.append(img.squeeze(0))
        infections.append(gt.squeeze(0))
    lungs = np.array(lungs)
    infections = np.array(infections)
#分法大小1数量不一致不想交
    train_data1,train_data2, train_label1,train_label2 = train_test_split(lungs, infections, test_size=0.9,
                                                                        random_state=42)
    data1_client=[(train_data1,train_label1),(train_data2,train_label2)]
    return data1_client

def crete_client4(path,size):
    lungs=[]
    infections=[]
    all_files = extract_files(path)
    for idx in range(len(all_files)):
        file_path = all_files[idx]
        mask_path = change_img_to_label_path1(file_path)

        slice = np.load(file_path)
        gt = np.load(mask_path)
        img = normalize(slice)  # 标准化和去除肺部阴影操作
        img = cv2.resize(img, (size, size))
        gt = cv2.resize(gt, (size, size), interpolation=cv2.INTER_NEAREST)

        lungs.append(img)
        infections.append(gt)
            #targets.append(target)
    lungs = np.array(lungs)
    infections = np.array(infections)

    #分法大小1数量不一致不相交
    train_data1,train_data2,train_label1,train_label2 = train_test_split(lungs, infections, test_size=0.8,
                                                                        random_state=42)
    data1_client=[(train_data1,train_label1),(train_data2,train_label2)]
    return data1_client

MIN_BOUND = -100.0
MAX_BOUND = 400.0
def normalize(image):
    """
    Perform standardization/normalization, i.e. zero_centering and setting
    the data to unit variance.
    """
    #image = setBounds(image, MIN_BOUND, MAX_BOUND)

    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    # image = np.clip(image, 0., 1.)
    # image = image - PIXEL_MEAN
    # image = image/PIXEL_STD
    return image

def crete_client2(path,size):
    lungs=[]
    infections=[]

    for i in range(len(path)):
        all_files = extract_files(path[i])

        for idx in range(len(all_files)):
            file_path = all_files[idx]
            mask_path = change_img_to_label_path1(file_path)

            slice = np.load(file_path)
            gt = np.load(mask_path)
            img = normalize(slice)  # 标准化和去除肺部阴影操作
            img = cv2.resize(img, (size, size))
            gt = cv2.resize(gt, (size, size), interpolation=cv2.INTER_NEAREST)

            lungs.append(img)
            infections.append(gt)

    return np.array(lungs),np.array(infections)


def crete_client3(path, size):
    lungs = []
    infections = []

    for i in range(len(path)):
        all_files = extract_files(path[i])

        for idx in range(len(all_files)):
            file_path = all_files[idx]
            mask_path = change_img_to_label_path1(file_path)

            slice = np.load(file_path)
            gt = np.load(file_path)
            img = normalize(slice)  # 标准化和去除肺部阴影操作
            img = cv2.resize(img, (size, size))
            gt = cv2.resize(gt, (size, size), interpolation=cv2.INTER_NEAREST)

            lungs.append(img)
            infections.append(gt)
                # targets.append(target)
            # lungs = np.array(lungs)
            # infections = np.array(infections)

    return np.array(lungs), np.array(infections)



#删除mask与data不匹配的切片

def del_mask():

    for i in range(450):
        file = str(i) + ".npy"
        path=os.path.join('../covid-19_data/Preprocessed/covid19-9/val/8/data/',file)
        print(path)
        if(not os.path.exists(path)):
            path2 = os.path.join('../covid-19_data/Preprocessed/covid19-9/val/8/masks/', file)
            print(path2+"删除完成")
            os.remove(path2)

#检测data和mask切片是否一一匹配
def match(path1):
    j=0
    for i in range(0, 300):
        file = str(i) + ".npy"
        slice1 = Path(path1 / "data" / file)
        mask1 = Path(path1 / "masks" / file)
        if(not os.path.exists(slice1)):
            print(str(slice1)+'不存在')
        else:
            if(os.path.exists(slice1) and os.path.exists(mask1)):
                j+=1
                slice = np.load(path1 / "data" / file)
                mask = np.load(path1 / "masks" / file)
                print(str(slice1)+'存在')
                print(slice.shape)
        print(j)
#将检测到的图片不符合尺寸的重新resize
def resize(path):
    path1 = Path("../covid-19_data/Preprocessed/covid19-ct-scans/train/5")
    j=0
    for i in range(0, 300):
        file = str(i) + ".npy"
        slice1 = Path(path1 / "data" / file)
        mask1 = Path(path1 / "masks" / file)
        if (not os.path.exists(slice1)):
            print(str(slice1) + '不存在')
        else:
            if (os.path.exists(slice1) and os.path.exists(mask1)):
                j += 1
                print(str(slice1) + '存在')
                slice = np.load(path1 / "data" / file)
                mask = np.load(path1 / "masks" / file)
                slice = cv2.resize(slice, (256, 256))
                mask = cv2.resize(mask, (256, 256))
                print(slice.shape,mask.shape)
                np.save(slice1,slice)
                np.save(mask1,mask)
                print(str(slice1)+"已resize")
        print(j)
def test_size(path):
    files = []
    for subject in path.glob("*"):  # Iterate over the subjects
        slice_path = subject / "data"  # Get the slices for current subject
        for slice in slice_path.glob("*"):
            files.append(slice)
    print(len(files))
    for idx in range (len(files)):
        print(files[idx])
        file_path = files[idx]
        mask_path = change_img_to_label_path1(file_path)
        slice = np.load(file_path)
        gt = np.load(file_path)

        print(slice.shape,gt.shape)


#展示切片
def plot(epoch,img,gt,flag):
    # Plot everything

    for i in range(len(img)):
        slice=img[i]
        mask=gt[i]  #[1,256,256]
        #out = mask * 255 #[1,256,256]
        # print(out.shape) [1,256,256]
        #out = (out).astype(np.uint8)
        # print(out.shape) [1,256,256]
        #out = out[0]
        #print(slice[0].shape)
        #print(out.shape)
        # a = Image.fromarray(np.uint8(out))
        #print(slice.shape,slice[0].shape)
        fig, axis = plt.subplots(1, 3, figsize=(4, 4))
        if (flag == 0):
            fig.suptitle('gt-plot')
            axis[0].imshow(slice[0], cmap="bone")
            axis[1].imshow(mask[0], cmap="bone")
            mask_ = np.ma.masked_where(mask[0] == 0, mask[0])
            axis[2].imshow(slice[0], cmap="bone")
            axis[2].imshow(mask_[0], cmap="autumn")
            #plt.show()
            plt.savefig('./result/epoch-' + str(epoch) +'-'+ str(i) + 'gt-plot.jpg')
        else:
            fig.suptitle('pred-plot')
            axis[0].imshow(slice[0], cmap="bone")
            axis[1].imshow(mask[0], cmap="bone")
            mask_ = np.ma.masked_where(mask == 0, mask)
            axis[2].imshow(slice[0], cmap="bone")
            axis[2].imshow(mask_[0], cmap="autumn")

            #plt.show()
            plt.savefig('./result/epoch-' + str(epoch) +'-'+ str(i) + 'pred-plot.jpg')

def plot_show(fed_round,client,img,gt,pred,mdice,flag):
    size=256
    for i in range(len(gt)):
        slice = gt[i]  #[1,256,256]
        image =img[i]#[1,256,256]
        #将mask的两个病变分开
        imgLr = pred[i].reshape(-1, 2, size, size)  # [1,2,256,256]

        imgM1r, imgM2r = imgLr[ :,0, :, :], imgLr[:,1, :, :]

        #print(slice.shape,imgM1r.shape) [1,256,256]
        # 预测合并
        imgRr = mergeChannels(imgLr, size) # [1,2,256,256]

        mask=imgRr.reshape(size, size)
        #print(imgLr.shape,mask.shape)

        fig, axis = plt.subplots(2, 3, figsize=(8, 8))
        fig.suptitle('round'+str(fed_round)+'-'+'slice'+str(i)+'-'+'Dice'+str(mdice))
        #backgroud = np.zeros((256, 256))
       # plt.imshow(backgroud, cmap=plt.cm.binary)


        mask_ = np.ma.masked_where(slice == 0, slice)
        pred_mask1_=np.ma.masked_where(imgM1r == 0, imgM1r)
        pred_mask2_ = np.ma.masked_where(imgM2r== 0, imgM2r)
        axis[0][0].imshow(image[0], cmap="bone")
        axis[0][0].imshow(mask_[0], cmap="autumn")

        axis[0][1].imshow(slice[0], cmap="bone")

        axis[0][2].imshow(mask, cmap="bone")


        axis[1][0].imshow(image[0], cmap="bone")
        axis[1][0].imshow(mask_[0], cmap="autumn")

        axis[1][1].imshow(image[0], cmap="bone")
        axis[1][1].imshow(mask_[0], cmap="autumn")

        axis[1][2].imshow(image[0], cmap="bone")
        axis[1][2].imshow(pred_mask1_[0], cmap="autumn")
        axis[1][2].imshow(pred_mask2_[0], cmap="summer_r")
        if(flag==0):
            # 保存本地测试的图片
            plt.savefig('./result/Local/round-{}-client-{}-{}-gt-pred.jpg'.format(fed_round,client,i))
        else:
            # 保存全局测试的图片
            plt.savefig('./result/Global/round-{}-client-{}-{}-gt-pred.jpg'.format(fed_round,client,i))


#定义函数就算移动平均损失值
def moving_average(a,w=10):
    #对损失值进行平滑处理，返回损失值的移动平均值
    if(len(a)<w):
        return a[:]
    return [val if idx<w else sum(a[(idx-w):idx])/w for idx , val in enumerate(a)]

#plot_loss
def plot_loss(losses):
    avgloss=moving_average(losses[0])
    avgloss1 = moving_average(losses[1])
    avgloss2 = moving_average(losses[2])
    avgloss3 = moving_average(losses[3])

    plt.plot(range(len(avgloss)),avgloss,'b--',color='g')
    plt.plot(range(len(avgloss1)), avgloss1, 'b--',color='r')
    plt.plot(range(len(avgloss2)), avgloss2, 'b--', color='y')
    plt.plot(range(len(avgloss3)), avgloss3, color='y')
    plt.xlabel('epochs')
    plt.ylabel('Dice')
    plt.legend(['10%_fedAvg', '20%_fedAvg', 'fully_fedAvg', 'fully_fedMixGAN'])
    plt.xlim(xmin=0, xmax=100)
    plt.show()

def zscore_normalize(x):
    y = (x - x.mean()) / x.std()
    return y

def covid_ked():
    # 以下是一些绘制图像的代码
    # 1.绘制数据的直方图
    # 绘制直方图
    # 构建数据
    # 计算数据集的平均值
    file_path = Path("../covid-19_data/Preprocessed/covid19-ct-scans/train/0/data/90.npy")
    slice = np.load(file_path)
    slice = zscore_normalize(slice)
    # print(slice.shape)  # (630,630)
    # 将医学图像数据转换为一维数组
    flat_data = slice.flatten()

    file_path1 = Path("../covid-19_data/Preprocessed/covid19-40/train/0/data/0.npy")
    slice1 = np.load(file_path1)
    slice1 = zscore_normalize(slice1)
    flat_data1 = slice1.flatten()

    file_path2 = Path("../covid-19_data/Preprocessed/covid19-9/train/0/data/4.npy")
    slice2 = np.load(file_path2)
    slice2 = zscore_normalize(slice2)
    flat_data2 = slice2.flatten()

    file_path3 = Path("../covid-19_data/Preprocessed/covid19_1110/train/0/data/11.npy")
    slice3 = np.load(file_path3)
    slice3 = zscore_normalize(slice3)
    flat_data3 = slice3.flatten()
    # print(cov)

    """
       绘制基本的单变量密度曲线图
       """
    # 设置绘图风格为'white'去掉网线格
    sns.set_style("white")
    plt.xlim(-1, 1)
    sns.kdeplot(flat_data, label='COVID-19-CT')
    sns.kdeplot(flat_data1, label='MSCOVID-19-CT')
    sns.kdeplot(flat_data2, label='COVID-19-9')
    sns.kdeplot(flat_data3, label='COVID-19-1110')
    plt.xlabel('Pixel intensity(z-score normalized)')
    plt.ylabel('Density')
    plt.legend()  # 显示图例
    plt.show()


def protaste_ked():
    # 以下是一些绘制图像的代码
    # 1.绘制数据的直方图
    # 绘制直方图
    # 构建数据
    # 计算数据集的平均值
    file_path = Path("../../prostate/Preprocessed/BIDMC/train/0/data/12.npy")
    slice = np.load(file_path)
    slice = zscore_normalize(slice)
    # print(slice.shape)  # (630,630)
    # 将医学图像数据转换为一维数组
    flat_data = slice.flatten()

    file_path1 = Path("../../prostate/Preprocessed/HK/train/0/data/5.npy")
    slice1 = np.load(file_path1)
    slice1 = zscore_normalize(slice1)
    flat_data1= slice1.flatten()

    file_path2 = Path("../../prostate/Preprocessed/UCL/train/0/data/8.npy")
    slice2 = np.load(file_path2)
    slice2 = zscore_normalize(slice2)
    flat_data2 = slice2.flatten()

    file_path3 = Path("../../prostate/Preprocessed/I2CVB/train/0/data/10.npy")
    slice3 = np.load(file_path3)
    slice3 = zscore_normalize(slice3)
    flat_data3 = slice3.flatten()

    file_path4 = Path("../../prostate/Preprocessed/BMC/train/0/data/3.npy")
    slice4 = np.load(file_path4)
    slice4 = zscore_normalize(slice4)
    flat_data4 = slice4.flatten()

    file_path5 = Path("../../prostate/Preprocessed/RUNMC/train/0/data/0.npy")
    slice5= np.load(file_path5)
    slice5 = zscore_normalize(slice5)
    flat_data5= slice5.flatten()


    # print(cov)

    """
       绘制基本的单变量密度曲线图
       """

    # 设置绘图风格为'white'去掉网线格
    sns.set_style("white")  # 设置横坐标显示范围
    plt.xlim(-2, 2)
    sns.kdeplot(flat_data, label='BIDMC')
    sns.kdeplot(flat_data1, label='HK')
    sns.kdeplot(flat_data2, label='UCL')
    sns.kdeplot(flat_data3, label='I2CVB')
    sns.kdeplot(flat_data4, label='BMC')
    sns.kdeplot(flat_data5, label='RUNMC')
    plt.xlabel('Pixel intensity(z-score normalized)')
    plt.ylabel('Density')
    plt.legend()  # 显示图例
    plt.show()
#将所有的切片展示出来，然后去掉没有注释的切片
def plot1():
    path = Path("../covid-19_data/Preprocessed/covid19_1110/test")
    all_files = extract_files(path)
    for i, path_file in enumerate(all_files):
        # 提取文件名
        slice_name = path_file.name
        print(i, path_file)
        mask_path = change_img_to_label_path1(path_file)
        slice = np.load(path_file)
        mask=np.load(mask_path)
        fig, axis = plt.subplots(1, 2, figsize=(4, 4))
        axis[0].imshow(slice, cmap="bone")
        mask_ = np.ma.masked_where(mask == 0, mask)
        axis[1].imshow(slice, cmap="bone")
        axis[1].imshow(mask_, cmap="autumn")
        plt.title('{}'.format(slice_name))
        plt.show()

#将dataset展现出来
def plot_dataset(train_loader):

    for slice,target, gt ,_,_,_ in train_loader:
        batch_image=[]
        batch_target=[]
        #print(len(slice))
        for i in range(0,4):
            #print(slice[i].shape, gt[i].shape)
            #slice,gt=slice[i].numpy(),gt[i].numpy()
            #print(slice[i].shape, gt[i].shape)
            image,mask1,gt1=augment(slice[i].numpy(),gt[i].numpy())
            image, mask1, gt1 = torch.from_numpy(image), torch.from_numpy(mask1), torch.from_numpy(gt1)
            print(image.shape,mask1.shape,gt1.shape)
            #fig, axis = plt.subplots(1, 2, figsize=(4, 4))
            #axis[0].imshow(slice[i][0], cmap="bone")
            #mask_ = np.ma.masked_where(gt[i] == 0, gt[i])
            #axis[0].imshow(mask_[0], cmap="autumn")
            #axis[1].imshow(image[0], cmap="bone")
            #mask1_ = np.ma.masked_where(gt1 == 0, gt1)
            #axis[1].imshow(mask1_[0], cmap="autumn")
        #axis[1].imshow(mask_[0], cmap="autumn")
            #plt.title('')
            #plt.show()
            batch_image.append(image)
            batch_target.append(mask1)

        #对图片做合并
        #for i in range(len(batch_image)):
        a=torch.unsqueeze(batch_image[0], dim=0)
        b =torch.unsqueeze(batch_image[1], dim=0)
        c =torch.unsqueeze(batch_image[2], dim=0)
        d =torch.unsqueeze(batch_image[3], dim=0)
        image1=torch.cat((a, b,c,d), 0)
        print(image1.shape)

#将dataset展现出来
def plot_dataset1(traindataset):

    for slice,target, gt,image_w,image_s in traindataset:
        batch_image=[]
        batch_target=[]
        #print(len(slice))
        # print(slice.shape, image_w.shape,image_s.shape)
        img = slice.squeeze(0)
        gt=gt.squeeze(0)
        image_w = image_w.squeeze(0)
        image_s = image_s.squeeze(0)
        print(img.shape, image_w.shape, image_s.shape)
        return img,gt,image_w,image_s

def change1(img,gt):
    # ------------------将图像做图像增强-------------------------
    img1,_,_=augment(img,gt)
    # ------------------生成并显示频谱图像-------------------------
    # numpy 傅里叶变换得到频谱,这个返回对象就是图像img的频谱信息，也是一个频谱图像，也叫频域图像。
    f = np.fft.fft2(img) #该频谱里面包含幅度和相位信息
    f1=np.fft.fft2(img1)
    #将零频率分量移动到中心位置， fshift也是一个浮点型的复数数组。fshift.shape返回(256,256)
    #fshift = np.fft.fftshift(f)
    #fshift1 = np.fft.fftshift(f1)
    # 获得幅度值
    fig_amp = np.abs(f)
    # 获得傅里叶变换的相位谱
    fig_pha = np.angle(f)

    fig1_amp =np.abs(f1)
    # 获得傅里叶变换的相位谱
    fig1_pha = np.angle(f1)
    # ------------------交换幅度信息-------------------------
    # exchange amplitude
    fig1_transform = fig1_amp * np.cos(fig_pha) + fig1_amp * np.sin(fig_pha) * 1j
    fig2_transform = fig_amp * np.cos(fig1_pha) + fig_amp * np.sin(fig1_pha) * 1j


    # -------------------------实现傅里叶逆变换-----------------------------------
#将频谱图像中的零频率分量移动到原位，原位就是频谱图像的左上角。



    img_back = np.fft.ifft2(fig1_transform) #逆傅里叶变换,变换后的结果还是一个复数数组
    img1_back = np.fft.ifft2(fig2_transform)
    #img_back是空域信息，但还是一个复数数组，需要将信息调整到[0,256]灰度空间内

    #u1=np.uint8(img_back)

    img_inv = np.abs(img_back)  #将复数数组转换到[0,256]灰度区间内
    img1_inv = np.abs(img1_back)

    return img_inv

    # ---------------------------可视化-------------------------------



def matplot_dataset(loss_dice):
    x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
          21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
          41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
          61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
          81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]

    plt.plot(x1, loss_dice[0], color='g',linestyle='--')
    plt.plot(x1,loss_dice[1],color='r',linestyle='--')
    plt.plot(x1, loss_dice[2], color='y',linestyle='--')
    plt.plot(x1, loss_dice[3], color='b', linestyle='--')
    plt.plot(x1, loss_dice[4], color='c', linestyle='--')
    plt.plot(x1, loss_dice[5], color='m', linestyle='--')
    #plt.plot(x1, loss_dice[3], color='g')
    #plt.plot(x1, loss_dice[4], color='r')
    #plt.plot(x1, loss_dice[5], color='y')
    plt.title('COVID-19-CT(client A)')  # 1.添加标题
    # plt.text(-2.5,30,'function y=x*x')#1.添加文字 第一、二个参数设置坐标
    # plt.annotate('这是一个示例注释',xy=(0,1),xytext=(-2,22),arrowprops={'headwidth':10,'facecolor':'r'})
    plt.xlabel('epoch')
    plt.ylabel('Dice')

    #plt.plot(x, x * x, color='r')
    #plt.plot(x, x * 3, color='0.5')
    #plt.plot(x, x * 4, color=(0.1, 0.2, 0.3))
    plt.legend(['FedAvg', 'FedProx','FedBN', 'MOON','FedNova','MixFedGAN'])
    # 调整x轴和y轴的格式
    #plt.locator_params('x', nbins=20)
    plt.xlim(xmin=0, xmax=100)
    plt.show()


def one_hot_segmentation(label, num_cls):
    batch_size = label.size(0)
    label = label.long()
    out_tensor = torch.zeros(batch_size, num_cls, *label.size()[2:]).to(label.device)
    out_tensor.scatter_(1, label, 1)

    return out_tensor


def dice_loss(pred, gt):
    # dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    assert pred.size() == gt.size(), "Input sizes must be equal."

    assert pred.dim() == 4, "Input must be a 4D Tensor."

    num = pred * gt
    num = torch.sum(num, dim=3)
    num = torch.sum(num, dim=2)

    den1 = pred * pred
    den1 = torch.sum(den1, dim=3)
    den1 = torch.sum(den1, dim=2)

    den2 = gt * gt
    den2 = torch.sum(den2, dim=3)
    den2 = torch.sum(den2, dim=2)

    epsilon = 1e-8  # Or another small value
    dice = 2 * (num / (den1 + den2 + epsilon))

    dice_total = 1 - torch.sum(dice) / dice.size(0)  # divide by batchsize

    return dice_total


def loss_dice(im1, im2, empty_score=1.0):
    """Calculates the dice coefficient for the images"""
    im1 = im1.data.cpu()
    im2 = im2.data.cpu()
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im1 = im1 > 0.5
    im2 = im2 > 0.5

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    # print(im_sum)
    dice = 2. * intersection.sum() / im_sum
    # print('LossDice: {:.4f}'.format(dice))
    dice=torch.from_numpy(dice)
    print(dice.shape)
    dice_total = 1 - dice / dice.size(0)
    return dice_total


def dice_coef_2d(pred, target):
    pred = torch.argmax(pred, dim=1, keepdim=True).float()
    target = torch.gt(target, 0.5).float()
    n = target.size(0)
    smooth = 1e-4

    target = target.view(n, -1)
    pred = pred.view(n, -1)
    intersect = torch.sum(target * pred, dim=-1)
    dice = (2 * intersect + smooth) / (torch.sum(target, dim=-1) + torch.sum(pred, dim=-1) + smooth)
    dice = torch.mean(dice)

    return dice


def dice(gt, pred):
    threshold = 0.5
    pred = (pred > threshold).astype(np.float)

    intersect = (pred * gt).sum()
    union = (pred + gt).sum()
    smooth = 1e-4
    return ((2.0 * intersect + smooth) / union + smooth)


def iou(gt, pred):
    threshold = 0.5
    pred = (pred > threshold).astype(np.float)

    intersect = (pred * gt).sum()
    union = (pred + gt).sum()
    smooth = 1e-4
    return (intersect + smooth / (union - intersect) + smooth)

def oversamples(train_dataset):
    target_list = []
    for slcie, label in tqdm(train_dataset):
        # Check if mask contains a tumorous pixel:
        if np.any(label):
            target_list.append(1)
        else:
            target_list.append(0)

    uniques = np.unique(target_list, return_counts=True)
    fraction = uniques[1][0] / uniques[1][1]
    print(fraction)
    weight_list = []
    for target in target_list:
        if target == 0:
            weight_list.append(1)
        else:
            weight_list.append(fraction)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weight_list, len(weight_list))

    return sampler

def augment(img,gt ):
    #print("已更新")
    #print(img.shape, gt.shape)
    #squeeze 函数：从数组的形状中删除单维度条目，即把shape中为1的维度去掉
    img=img.cpu()
    gt=gt.cpu()
    img=img.detach().numpy()
    gt=gt.detach().numpy()
    #gt=gt[:,1:,:,:]
    #img = np.squeeze(img, 1)
    #gt=np.squeeze(gt,1)
    seq = iaa.Sequential([
            iaa.Affine(translate_percent=(0.15),
                       scale=(0.85, 1.15),  # zoom in or out
                       rotate=(-45, 45)  #
                       ),  # rotate up to 45 degrees
            iaa.ElasticTransformation()  # Elastic Transformations
        ])
        ###################IMPORTANT###################
        # Fix for https://discuss.pytorch.org/t/dataloader-workers-generate-the-same-random-augmentations/28830/2
    random_seed = torch.randint(0, 1000000, (1,))[0].item()
    imgaug.seed(random_seed)
    #####################################################
    batch_image = []
    batch_target = []
    # Augment images and segmaps.
    for i in range(len(img)):
        mask = SegmentationMapsOnImage(gt[i], gt[i].shape)
        slice_aug, mask_aug1=seq(image=img[i], segmentation_maps=mask[0:,:,:])
        print(slice_aug.shape,mask_aug1.shape)
        mask_aug1 = mask_aug1.get_arr()
        mask_aug1,mask_aug2=seq(image=mask_aug1, segmentation_maps=mask[0:, :, :])
        mask_aug2=mask_aug2.get_arr()
        mask1 = np.vstack((np.asarray(mask_aug1), np.asarray(mask_aug2)))
        print(slice_aug.shape,mask1.shape)

        slice_aug = torch.from_numpy(np.expand_dims(slice_aug,0))
        mask1 = torch.from_numpy(np.expand_dims(mask1,0))
        batch_image.append(slice_aug)
        batch_target.append(mask1)
    image1 = torch.cat(batch_image, 0)
    maskaa=torch.cat(batch_target, 0)
    print(image1.shape,maskaa.shape)
    return image1,maskaa

#计算
def compute_model_change(netS,netS_local):
    scorce1 = 0.0
    for w, w_t in zip(netS.parameters(), netS_local.parameters()):
        scorce1 += (w - w_t).norm(2).cpu()
    return scorce1


# def ampfigure2(img,img1):
#     # numpy 傅里叶变换得到频谱,这个返回对象就是图像img的频谱信息，也是一个频谱图像，也叫频域图像。
#     f = np.fft.fft2(img)  # 该频谱里面包含幅度和相位信息
#     f1 = np.fft.fft2(img1)
#     # 将零频率分量移动到中心位置， fshift也是一个浮点型的复数数组。fshift.shape返回(256,256)
#     fshift = np.fft.fftshift(f)
#     fshift1 = np.fft.fftshift(f1)
#     # 获得幅度值
#     fig_amp = np.abs(fshift)
#     # 获得傅里叶变换的相位谱
#     fig_pha = np.angle(fshift)
#
#     fig1_amp = np.abs(fshift1)
#     # 获得傅里叶变换的相位谱
#     fig1_pha = np.angle(fshift1)
#     # ------------------交换幅度信息-------------------------
#     amp=low_freq_mutate_np(fig_amp, fig1_amp, L=0.1)
#     # exchange amplitude
#     fig1_transform=amp * np.exp(1j * fig_pha)
#     fig2_transform = amp * np.exp(1j * fig1_pha)
#     #fig1_transform = fig1_amp * np.cos(fig_pha) + fig1_amp * np.sin(fig_pha) * 1j
#     #fig2_transform = fig_amp * np.cos(fig1_pha) + fig_amp * np.sin(fig1_pha) * 1j
#
#     # -------------------------实现傅里叶逆变换-----------------------------------
#     # 将频谱图像中的零频率分量移动到原位，原位就是频谱图像的左上角。
#
#     img_back = np.fft.ifft2(fig1_transform)  # 逆傅里叶变换,变换后的结果还是一个复数数组
#     img1_back = np.fft.ifft2(fig2_transform)
#     # img_back是空域信息，但还是一个复数数组，需要将信息调整到[0,256]灰度空间内
#
#     # u1=np.uint8(img_back)
#
#     img_inv = np.abs(img_back)  # 将复数数组转换到[0,256]灰度区间内
#     img1_inv = np.abs(img1_back)
#     # 移位到中点的频率图
#     plt.figure(figsize=(20, 8))
#     plt.subplot(181), plt.imshow(img, cmap='gray')  # 图像A原图
#     plt.subplot(182), plt.imshow(fig_amp, cmap='gray')  # 图像A的幅度
#     plt.subplot(183), plt.imshow(fig_pha, cmap='gray')  # 图像A的相位
#     plt.subplot(184), plt.imshow(img_inv, cmap='gray')  # 图像A相位+图像B幅度
#     plt.subplot(185), plt.imshow(img1, cmap='gray')  # 图像B原图
#     plt.subplot(186), plt.imshow(fig1_amp, cmap='gray')  # 图像B的幅度
#     plt.subplot(187), plt.imshow(fig1_pha, cmap='gray')  # 图像B的相位
#     plt.subplot(188), plt.imshow(img1_inv, cmap='gray')  # 图像B相位+图像A幅度
#     plt.show()
#
# # def ampfiguere1(img):
# #     result = dft(img)
# #     # 将傅里叶频谱图从左上角移动到中心位置
# #     fshift = np.fft.fftshift(result)
# #     # 将复数转为浮点数进行傅里叶频谱图显示
# #     fimg = np.log(np.abs(fshift))
# #     # 调用傅里叶逆变换函数
# #     result2 = idft(result)
# #
# #     # ---------------------------分离高低频信号-------------------------------
# #     center_row = int(img.shape[0] / 2)  # 生成中心点
# #     center_col = int(img.shape[1] / 2)
# #
# #     fshift1 = fshift.copy()
# #     fshift1[center_row - 30:center_row + 30, center_col - 30:center_col + 30] = 0  # 把中间的区域置为0，生成高频信号频域图
# #     ifshift1 = np.fft.ifftshift(fshift1)
# #     iimg = np.fft.ifft2(ifshift1)
# #     img_inv1 = np.abs(iimg)
# #
# #     fshift2 = fshift.copy()  # 把四周区域都置为0，生成低频信号频域图
# #     fshift2[0:center_row - 30] = 0
# #     fshift2[center_row + 30:] = 0
# #     fshift2[:, 0:center_col - 30] = 0
# #     fshift2[:, center_col + 30:] = 0
# #     ifshift2 = np.fft.ifftshift(fshift2)
# #     iimg2 = np.fft.ifft2(ifshift2)
# #     img_inv2 = np.abs(iimg2)
# #     # 图像显示
# #     plt.figure(figsize=(16, 6))
# #     plt.subplot(161), plt.imshow(img,cmap='gray'), plt.title('原图像')
# #     plt.subplot(162), plt.imshow(fimg,cmap='gray'), plt.title('傅里叶变换')
# #     plt.subplot(163), plt.imshow(result2,cmap='gray'), plt.title('傅里叶逆变换')
# #     plt.subplot(164), plt.imshow(img_inv1, cmap='gray'), plt.title('高频')
# #     plt.subplot(165), plt.imshow(img_inv2, cmap='gray'), plt.title('低频')
# #
# #     plt.show()


def ampfigure(img):
    # ------------------生成并显示频谱图像-------------------------------------
    f = np.fft.fft2(img)  # 生成频谱图像, f是一个浮点型的数组，是一个复数数组。  f.shape返回(512, 512)
    fshift = np.fft.fftshift(f)  # 将零频率分量移动到中心位置， fshift也是一个浮点型的复数数组。fshift.shape返回(512, 512)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))  # 为显示图像，将复数值调到[0,256]的灰度空间内。  magnitude_spectrum.shape返回(512, 512)
    magnitude_spectrum1 = 20 * np.log(np.abs(f))  # 假如不移动零频率分量，看看图像呈现什么样子

    # -------------------------实现傅里叶逆变换-----------------------------------
    ishift = np.fft.ifftshift(fshift)  # 将零频率分量还原      (ishift == f).sum() = 512*512
    iimg = np.fft.ifft2(ishift)  # 逆傅里叶变换,变换后的结果还是一个复数数组
    img_inv = np.abs(iimg)  # 将复数数组转换到[0,256]灰度区间内

    # ---------------------------分离高低频信号-------------------------------
    center_row = int(img.shape[0] / 2)  # 生成中心点
    center_col = int(img.shape[1] / 2)

    fshift1 = fshift.copy()
    fshift1[center_row - 30:center_row + 30, center_col - 30:center_col + 30] = 0  # 把中间的区域置为0，生成高频信号频域图
    ifshift1 = np.fft.ifftshift(fshift1)
    iimg = np.fft.ifft2(ifshift1)
    img_inv1 = np.abs(iimg)

    fshift2 = fshift.copy()  # 把四周区域都置为0，生成低频信号频域图
    fshift2[0:center_row - 30] = 0
    fshift2[center_row + 30:] = 0
    fshift2[:, 0:center_col - 30] = 0
    fshift2[:, center_col + 30:] = 0
    ifshift2 = np.fft.ifftshift(fshift2)
    iimg2 = np.fft.ifft2(ifshift2)
    img_inv2 = np.abs(iimg2)

    # 可视化：
    plt.figure(figsize=(16, 6))
    plt.subplot(161), plt.imshow(img, cmap='gray')  # 原图
    plt.subplot(162), plt.imshow(magnitude_spectrum, cmap='gray')  # 频域图
    plt.subplot(163), plt.imshow(magnitude_spectrum1, cmap='gray')  # 不移动零频率分量的频域图
    plt.subplot(164), plt.imshow(img_inv, cmap='gray')  # 频域图逆变换的结果
    plt.subplot(165), plt.imshow(img_inv1,
                                 cmap='gray')  # 高频信号频域图的逆傅里叶变换结果图，都是边缘，都是像素变换较大的区域。比如脸里面、帽子里面、皮肤里面变化较小的区域都过滤掉了。
    plt.subplot(166), plt.imshow(img_inv2, cmap='gray')  # 低频信号频域图的逆傅里叶变换结果图，边缘模糊了，都是一些变换较缓的区域。和前面图的情况正好相反。
    plt.show()


# 混淆矩阵定义
def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def plot_maxtrix(maxtrix, per_kinds):
    # 分类标签
    lables = ['Infection', 'NotInfection']

    Maxt = np.empty(shape=[0, 2])

    m = 0
    for i in range(2):
        print('row sum:', per_kinds[m])
        f = (maxtrix[m, :] * 100) / per_kinds[m]
        Maxt = np.vstack((Maxt, f))
        m = m + 1

    thresh = Maxt.max() / 1

    plt.imshow(Maxt, cmap=plt.cm.Blues)

    for x in range(2):
        for y in range(2):
            info = float(format('%.1f' % F[y, x]))
            print('info:', info)
            plt.text(x, y, info, verticalalignment='center', horizontalalignment='center')
    plt.tight_layout()
    plt.yticks(range(2), lables)  # y轴标签
    plt.xticks(range(2), lables, rotation=45)  # x轴标签
    plt.savefig('./test.png', bbox_inches='tight')  # bbox_inches='tight'可确保标签信息显示全
    plt.show()


import numpy as np


def calc_semantic_segmentation_confusion(pred_labels, gt_labels):
    pred_labels = iter(pred_labels)
    gt_labels = iter(gt_labels)

    n_class = 2
    confusion = np.zeros((n_class, n_class), dtype=np.int64)  # (12, 12)

    for pred_label, gt_label in zip(pred_labels, gt_labels):



        if pred_label.shape != gt_label.shape:
            raise ValueError('Shape of ground truth and prediction should be same.')

        pred_label = pred_label.flatten()  # liupc:我感觉这两句多余
        gt_label = gt_label.flatten()

        # Dynamically expand the confusion matrix if necessary.
        lb_max = np.max((pred_label, gt_label))
        # print(lb_max)
        if lb_max >= n_class:  # 如果分类数大于预设的分类数目，则扩充一下。
            expanded_confusion = np.zeros((lb_max + 1, lb_max + 1), dtype=np.int64)
            expanded_confusion[0:n_class, 0:n_class] = confusion

            n_class = lb_max + 1
            confusion = expanded_confusion

        # Count statistics from valid pixels
        mask = gt_label >= 0
        confusion += np.bincount(
            n_class * gt_label[mask].astype(int) + pred_label[mask],
            minlength=n_class ** 2).reshape((n_class, n_class))

    for iter_ in (pred_labels, gt_labels):
        # This code assumes any iterator does not contain None as its items.
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same')

    return confusion


def show_sparsity(model, save_name, model_path=None):


    # get all scaler parameters form the network:
    scaler_list = []
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d) and m.weight is not None:
            m_cpu = m.weight.data.cpu().numpy().squeeze()
            # print('m_cpu:', type(m_cpu), m_cpu.shape)
            scaler_list.append(m_cpu)
    all_scaler = np.concatenate(scaler_list, axis=0)
    print('all_scaler:', all_scaler.shape, 'L0 (sum):', np.sum(all_scaler != 0), 'L1 (mean):',
          np.mean(np.abs(all_scaler)))

    # save npy and plt png:
    # np.save(save_name + '.npy', all_scaler)
    n, bins, patches = plt.hist(all_scaler, 50)
    # print(n)
    plt.savefig(save_name + '.png')
    plt.close()

    return all_scaler

class CriterionKD(nn.Module):
    '''
    knowledge distillation loss
    '''

    def __init__(self, upsample=False):
        super(CriterionKD, self).__init__()
        self.upsample = upsample
        self.criterion_kd = torch.nn.KLDivLoss()

    def forward(self, pred, soft,temperature=10):
        soft.detach()
        scale_pred = pred
        scale_soft = soft
        loss =self.criterion_kd(torch.log_softmax(scale_pred / temperature, dim=1),
            torch.softmax(scale_soft / temperature, dim=1)
        ) * (temperature ** 2)
        return loss
