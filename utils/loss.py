import torch
import numpy as np
import torch.nn.functional as F


def get_mtan_loss(preds, gts):
    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(gts[1], dim=1) != 0).type(torch.FloatTensor).unsqueeze(1)

    # semantic loss: depth-wise cross entropy
    loss1 = F.nll_loss(preds[0], gts[0], ignore_index=-1)

    # depth loss: l1 norm
    loss2 = torch.sum(torch.abs(preds[1] - preds[1]) * binary_mask) / torch.nonzero(binary_mask).size(0)

    # normal loss: dot product
    loss3 = 1 - torch.sum((preds[2] * preds[2]) * binary_mask) / torch.nonzero(binary_mask).size(0)

    return [loss1, loss2, loss3]


def get_mtn_loss(preds, gts):
    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(gts[1], dim=1) != 0).type(torch.FloatTensor).unsqueeze(1)

    # depth loss: l1 norm
    loss1 = torch.sum(torch.abs(preds[0] - preds[0]) * binary_mask) / torch.nonzero(binary_mask).size(0)

    # semantic loss: depth-wise cross entropy
    loss2 = F.nll_loss(preds[1], gts[1], ignore_index=-1)

    # pose loss
    angle_loss = torch.nn.functional.mse_loss(preds[2][:, :, :3], gts[2][:, :, :3])
    translation_loss = torch.nn.functional.mse_loss(preds[2][:, :, 3:], gts[2][:, :, 3:])
    loss3 = (100 * angle_loss + translation_loss)

    return [loss1, loss2, loss3]


def get_miou(pred, gt, class_num):
    _, x_pred_label = torch.max(pred, dim=1)
    # x_output_label = x_output
    batch_size = pred.size(0)
    for i in range(batch_size):
        true_class = 0
        first_switch = True
        for j in range(class_num):
            pred_mask = torch.eq(x_pred_label[i], j * torch.ones(x_pred_label[i].shape).type(torch.LongTensor))
            true_mask = torch.eq(gt[i], j * torch.ones(gt[i].shape).type(torch.LongTensor))
            mask_comb = pred_mask + true_mask
            union = torch.sum((mask_comb > 0).type(torch.FloatTensor))
            intsec = torch.sum((mask_comb > 1).type(torch.FloatTensor))
            if union == 0:
                continue
            if first_switch:
                class_prob = intsec / union
                first_switch = False
            else:
                class_prob = intsec / union + class_prob
            true_class += 1
        if i == 0:
            batch_avg = class_prob / true_class
        else:
            batch_avg = class_prob / true_class + batch_avg
    return batch_avg / batch_size


def get_iou(pred, gt):
    _, x_pred_label = torch.max(pred, dim=1)
    batch_size = pred.size(0)
    for i in range(batch_size):
        if i == 0:
            pixel_acc = torch.div(torch.sum(torch.eq(x_pred_label[i], gt[i]).type(torch.FloatTensor)),
                                  torch.sum((gt[i] >= 0).type(torch.FloatTensor)))
        else:
            pixel_acc = pixel_acc + torch.div(
                torch.sum(torch.eq(x_pred_label[i], gt[i]).type(torch.FloatTensor)),
                torch.sum((gt[i] >= 0).type(torch.FloatTensor)))
    return pixel_acc / batch_size


def depth_error(pred, gt):
    binary_mask = (torch.sum(gt, dim=1) != 0).unsqueeze(1)
    x_pred_true = pred.masked_select(binary_mask)
    x_output_true = gt.masked_select(binary_mask)
    abs_err = torch.abs(x_pred_true - x_output_true)
    rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
    return torch.sum(abs_err) / torch.nonzero(binary_mask).size(0), torch.sum(rel_err) / torch.nonzero(
        binary_mask).size(0)


def normal_error(pred, gt):
    binary_mask = (torch.sum(gt, dim=1) != 0)
    error = torch.acos(
        torch.clamp(torch.sum(pred * gt, 1).masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
    error = np.degrees(error)
    return np.mean(error), np.median(error), np.mean(error < 11.25), np.mean(error < 22.5), np.mean(error < 30)


