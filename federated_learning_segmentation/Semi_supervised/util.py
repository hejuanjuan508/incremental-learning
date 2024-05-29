
#渐增系数
import math
import torch
import torch.nn.functional as F
from random import random
from torch.autograd import Variable
import numpy as np


def cal_consistency_weight(epoch, init_ep=0, end_ep=150, init_w=0.0, end_w=20.0):
    """Sets the weights for the consistency loss"""
    if epoch > end_ep:
        weight_cl = end_w
    elif epoch < init_ep:
        weight_cl = init_w
    else:
        T = float(epoch - init_ep) / float(end_ep - init_ep)
        # weight_mse = T * (end_w - init_w) + init_w #linear
        weight_cl = (math.exp(-5.0 * (1.0 - T) * (1.0 - T))) * (end_w - init_w) + init_w  # exp
    # print('Consistency weight: %f'%weight_cl)
    return weight_cl



# calculate gradient penalty
def cal_gradient_penalty(netD,  real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
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
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2,
                                                      dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None
def semi_cbc_loss(inputs, targets,
                  threshold=0.6,
                  neg_threshold=0.3,
                  conf_mask=True):
    if not conf_mask:
        raise NotImplementedError
    targets_prob = F.softmax(targets, dim=1)
    # for positive
    pos_weight = targets_prob.max(1)[0]
    pos_mask = (pos_weight >= threshold)
    # for negative
    neg_weight = targets_prob.min(1)[0]
    neg_mask = (neg_weight < neg_threshold)
    y_tilde = torch.argmax(targets, dim=1)
    if not torch.any(pos_mask):
        positive_loss_mat = torch.tensor([.0], device=targets.device)
    else:
        positive_loss_mat = F.nll_loss(torch.log_softmax(inputs, dim=1),
                                       y_tilde, reduction="none")
        positive_loss_mat = positive_loss_mat * pos_weight
        positive_loss_mat = positive_loss_mat[pos_mask]

    if not torch.any(neg_mask):
        negative_loss_mat = torch.tensor([.0], device=targets.device)
    else:
        inverse_prob = torch.clamp(1 - F.softmax(inputs, dim=1), min=1e-6, max=1.0)
        negative_loss_mat = F.nll_loss(inverse_prob.log(), (1 - y_tilde), reduction="none")
        negative_loss_mat = negative_loss_mat * neg_weight
        negative_loss_mat = negative_loss_mat[neg_mask]

    return positive_loss_mat.mean() + negative_loss_mat.mean(), None

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes


#EMA
def update_ema_variables(model, model_teacher, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1.0 - 1.0 / float(global_step + 1), alpha)
    for param_t, param in zip(model_teacher.parameters(), model.parameters()):
        param_t.data.mul_(alpha).add_(1 - alpha, param.data)

#一个用来调节loss weights的函数
def get_current_consistency_weight(args, epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def kl_div_with_logit(q_logit, p_logit):

    q = F.softmax(q_logit, dim=1)
    logq = F.log_softmax(q_logit, dim=1)
    logp = F.log_softmax(p_logit, dim=1)

    qlogq = ( q *logq).sum(dim=1).mean(dim=0)
    qlogp = ( q *logp).sum(dim=1).mean(dim=0)

    return qlogq - qlogp
def _l2_normalize(d):

    d = d.numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3))).reshape((-1, 1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)
def vat_loss(model, ul_x, ul_y, xi, eps, num_iters):

    # find r_adv

    d = torch.Tensor(ul_x.size()).normal_()
    for i in range(num_iters):
        d = xi *_l2_normalize(d)
        d = Variable(d.cuda(), requires_grad=True)
        y_hat = model(ul_x + d)
        delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
        delta_kl.backward(delta_kl.clone().detach())

        d = d.grad.data.clone().cpu()
        model.zero_grad()

    d = _l2_normalize(d)
    d = Variable(d.cuda())
    r_adv = eps *d
    # compute lds
    y_hat = model(ul_x + r_adv.detach())
    delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
    return delta_kl

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

    dice = 2 * (num / (den1 + den2))

    dice_total = 1 - torch.sum(dice) / dice.size(0)  # divide by batchsize

    return dice_total
