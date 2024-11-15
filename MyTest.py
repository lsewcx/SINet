import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import imageio  # 使用 imageio 替代 scipy.misc.imsave
from Src.SINet import SINet_ResNet50
from Src.utils.Dataloader import test_dataset
from Src.utils.trainer import eval_mae, numpy2tensor

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='the snapshot input size')
parser.add_argument('--model_path', type=str,
                    default='./Snapshot/2020-CVPR-SINet/SINet_40.pth')
parser.add_argument('--test_save', type=str,
                    default='./Result/2020-CVPR-SINet-New/')
parser.add_argument('--image_root', type=str, default="/kaggle/input/cod10k-test/TestDataset/COD10K/Imgs/", help='the root directory of test images')
parser.add_argument('--gt_root', type=str, default="/kaggle/input/cod10k-test/TestDataset/COD10K/GT/", help='the root directory of ground truth images')
opt = parser.parse_args()

# 检测设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载模型
model = SINet_ResNet50().to(device)
model.load_state_dict(torch.load(opt.model_path, map_location=device))
model.eval()

# 初始化总的 MAE 和图像计数
total_mae = 0.0
total_images = 0

for dataset in ['COD10K']:
    save_path = opt.test_save + dataset + '/'
    os.makedirs(save_path, exist_ok=True)
    # NOTES:
    #  if you plan to inference on your customized dataset without ground-truth,
    #  you just modify the params (i.e., `image_root=your_test_img_path` and `gt_root=your_test_img_path`)
    #  with the same filepath. We recover the original size according to the shape of ground-truth, and thus,
    #  the ground-truth map is unnecessary actually.
    test_loader = test_dataset(image_root=opt.image_root,
                               gt_root=opt.gt_root,
                               testsize=opt.testsize)
    img_count = 1
    for iteration in range(test_loader.size):
        # 加载数据
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.to(device)
        # 推理
        _, cam = model(image)
        # 调整大小并压缩
        cam = F.interpolate(cam, size=gt.shape, mode='bilinear', align_corners=True)
        cam = cam.sigmoid().data.cpu().numpy().squeeze()
        # 归一化
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        # 保存为 .npy 文件
        np.save(save_path + name.replace('.png', '.npy'), cam)
        # 评估
        mae = eval_mae(torch.from_numpy(cam).to(device), torch.from_numpy(gt).to(device))
        total_mae += mae.item()
        total_images += 1
        # 粗略评分
        print('[Eval-Test] Dataset: {}, Image: {} ({}/{}), MAE: {}'.format(dataset, name, img_count,
                                                                           test_loader.size, mae))
        img_count += 1

# 计算并输出平均 MAE
average_mae = total_mae / total_images
print(f"\n[Summary] Average MAE: {average_mae:.4f} over {total_images} images")

print("\n[Congratulations! Testing Done]")