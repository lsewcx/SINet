import numpy as np
import matplotlib.pyplot as plt

def display_npy_image(file_path):
    # 读取 .npy 文件
    image_data = np.load(file_path)
    
    # 显示图片
    plt.imshow(image_data, cmap='gray')
    plt.colorbar()
    plt.title('Image from .npy file')
    plt.show()

# 示例调用
file_path = 'SINet/Result/2020-CVPR-SINet-New/COD10K/COD10K-CAM-1-Aquatic-1-BatFish-2.npy'  # 替换为你的 .npy 文件路径
display_npy_image(file_path)