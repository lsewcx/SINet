import torch


def dice_loss(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice

class HuberLoss(torch.nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, pred, target):
        abs_error = torch.abs(pred - target)
        quadratic = torch.min(abs_error, torch.tensor(self.delta).to(abs_error.device))
        linear = abs_error - quadratic
        loss = 0.5 * quadratic**2 + self.delta * linear
        return loss.mean()