import torch
from Src.SINet import SINet_ResNet50

model_SINet = SINet_ResNet50(channel=32)

input = torch.randn(1, 3, 352, 352)
output = model_SINet(input)
print(output[0].shape, output[1].shape)