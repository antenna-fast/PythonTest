import torch
from torchvision import transforms
from torch.nn import functional as F
import cv2
from PIL import Image

import matplotlib.pyplot as plt


img_path = '/mnt/data/yaohualiu/PublicDataset/wildtrack_dataset/Wildtrack_dataset/Image_subsets/C1/00000000.png'
img = Image.open(img_path)
img_torch = transforms.ToTensor()(img).unsqueeze(0)  # BS CH H W

plt.imshow(img_torch[0].numpy().transpose(1, 2, 0))  # show: H W C
plt.title('oriImg')
plt.show()

# 1. Affine matrix
# 需要给定affine matrix

# takes an affine transformation matrix and produces a set of sampling coordinates
# theta, size, align_corners=None
# input: batch of affine matrices with shape (N x 2 × 3) for 2D or (N x 3 x 4) for 3D
# return:

# 平移量 [-1, 1]  # 左+右- 上+下-
# theta = torch.tensor([[1, 0, -100],
#                       [0, 1, -300]], dtype=torch.float).unsqueeze(0)  # to batch = 1
theta = torch.tensor([[1, 0, 1],
                      [0, 1, 0]], dtype=torch.float).unsqueeze(0)
# grid = F.affine_grid(theta, img_torch.size(), align_corners=True)
grid = F.affine_grid(theta, img_torch.size())
# ref: https://zhuanlan.zhihu.com/p/87572724
# grid_size: [1, 1080, 1920, 2]

# samples the grid at those coordinates.
# Back-propagating gradients during the backward step is handled automatically by pyTorch.
output = F.grid_sample(img_torch, grid=grid)

new_img_torch = output[0]
plt.imshow(new_img_torch.numpy().transpose(1, 2, 0))
plt.show()

# 2. Homograph

