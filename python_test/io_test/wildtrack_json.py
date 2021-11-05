import json
import os

file_path = '00000000.json'
file = open(file_path)
anno = json.load(file)

# print(anno)
print(type(anno))
print(type(anno[0]))
print(anno[0])
print(anno[0].keys())
print(anno[0]['views'])  # 存在-1
