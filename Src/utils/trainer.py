import torch
from torch.autograd import Variable
from datetime import datetime
import os
import torch.nn.functional as F

def eval_mae(y_pred, y):
    """
    evaluate MAE (for test or validation phase)
    :param y_pred:
    :param y:
    :return: Mean Absolute Error
    """
    return torch.abs(y_pred - y).mean()

def numpy2tensor(numpy):
    """
    convert numpy_array in cpu to tensor in gpu
    :param numpy:
    :return: torch.from_numpy(numpy).cuda()
    """
    return torch.from_numpy(numpy).cuda()

def clip_gradient(optimizer, grad_clip):
    """
    recalibrate the misdirection in the training
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_lr(optimizer, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay

def trainer(train_loader, model, optimizer, epoch, opt, loss_func, total_step, best_mae, save_best_model):
    """
    Training iteration
    :param train_loader:
    :param model:
    :param optimizer:
    :param epoch:
    :param opt:
    :param loss_func:
    :param total_step:
    :param best_mae:
    :param save_best_model:
    :return: best_mae
    """
    model.train()
    for step, data_pack in enumerate(train_loader):
        optimizer.zero_grad()
        images, gts = data_pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()

        cam_sm, cam_im = model(images)
        loss_sm = loss_func(cam_sm, gts)
        loss_im = loss_func(cam_im, gts)
        loss_total = loss_sm + loss_im

        loss_total.backward()

        # clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if step % 10 == 0 or step == total_step:
            print('[{}] => [Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}/{:04d}] => [Loss_s: {:.4f} Loss_i: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, step, total_step, loss_sm.data, loss_im.data))

    save_path = opt.save_model
    os.makedirs(save_path, exist_ok=True)

    # 验证模型并保存最好的模型
    model.eval()
    total_mae = 0.0
    total_images = 0
    for step, data_pack in enumerate(train_loader):
        images, gts = data_pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()

        with torch.no_grad():
            cam_sm, cam_im = model(images)
            cam = (cam_sm + cam_im) / 2
            mae = eval_mae(cam, gts)
            total_mae += mae.item()
            total_images += 1

    average_mae = total_mae / total_images
    print(f'[Validation] Epoch: {epoch}, Average MAE: {average_mae:.4f}')

    if average_mae < best_mae:
        best_mae = average_mae
        save_best_model(model, save_path, epoch, best_mae)

    return best_mae

def save_best_model(model, save_path, epoch, best_mae):
    """
    Save the best model
    :param model:
    :param save_path:
    :param epoch:
    :param best_mae:
    :return:
    """
    model_save_path = os.path.join(save_path, f'SINet_best_{epoch}_MAE_{best_mae:.4f}.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f'Saved best model at epoch {epoch} with MAE {best_mae:.4f}')