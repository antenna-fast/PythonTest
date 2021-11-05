import keyword
import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/test_projector')  # 文件夹 / 不同的模块

meta = []
while len(meta) < 100:
    meta = meta + keyword.kwlist  # get some strings
meta = meta[:100]
print('meta is ', meta)  # 一堆字符串

for i, v in enumerate(meta):
    meta[i] = v + str(i)
print('meta ', meta)

label_img = torch.rand(100, 3, 10, 32)
for i in range(100):
    label_img[i] *= i / 100.0

# mat (torch.Tensor or numpy.array): A matrix which each row is the feature vector of the data point
# metadata (list): A list of labels, each element will be convert to string
# label_img (torch.Tensor): Images correspond to each data point
# global_step (int): Global step value to record
# tag (string): Name for the embedding

# writer.add_embedding(torch.randn(100, 5), metadata=meta, label_img=label_img)
writer.add_embedding(torch.randn(100, 5), label_img=label_img)
# writer.add_embedding(torch.randn(100, 5), metadata=meta)

# read feature
# writer.add_embedding()

writer.close()
