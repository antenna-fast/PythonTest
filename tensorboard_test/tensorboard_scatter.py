import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import numpy as np
from PIL import Image

import os

# Writer will output to ./runs/ directory by default
writer = SummaryWriter('runs/test_tboard')   # 文件夹 / 不同的模型
# writer = SummaryWriter()

# writer.add_image('images', grid, 0)
# writer.add_graph(model, images)

# writer.close()

# 性能参数 / 数据集

for n_iter in range(300):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

# img = Image.open('/Users/aibee/Desktop/11.png')
# writer.add_image('image/demo', img)  # tag, img_tensor

writer.close()
