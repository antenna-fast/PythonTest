import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# 单机单卡
# Pytorch will only use one GPU by default.
# device = torch.device("cuda:0")
# model.to(device)
# my_tensor = my_tensor.to(device)
# print("device_count is ", torch.cuda.device_count())  # default is 8
# print('device is ', device)  # cuda:0


# # 单机多卡
gpu_id = "1, 2, 3, 4"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id  # need to specify at first
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device is ', device)
# model = nn.DataParallel(model)  # 模型
# model.to(device)

# inputs = inputs.cuda()  # 数据
# labels = labels.cuda()


# 多机多卡(分布式)

# 多卡实例
# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100


class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)


class Model(nn.Module):
    # Our model
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output


model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)  # 关键

model.to(device)
# print(model.device_ids)  # 分配的gpu会重新排序 [0, 1, 2, 3]
# print(model)  # 单卡无法查看ids


for data in rand_loader:
    input = data.to(device)
    output = model(input)  # 自动将数据拆分到不同的model
    print("Outside: input size", input.size(),
          "output_size", output.size())
    print('one epoch ... ')
