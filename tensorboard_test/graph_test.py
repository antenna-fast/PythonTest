import torch
from torchvision.models import resnet50

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/test_graph')  # 文件夹 / 不同的模块

fake_data = torch.ones(1, 3, 10, 10)

model = resnet50()
print(model)

writer.add_graph(model, fake_data)
# writer.add_graph(model)

writer.close()
