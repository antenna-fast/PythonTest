import json

# Python 字典类型转换为 JSON 对象
data = {
    1: {'pid1': '0001',
        'image_id': 'img_id',
        'image_name': 'image_name'
        }

}

json_str = json.dumps(data)
# print("Python 原始数据：", repr(data))
print("JSON 对象：", json_str)

# print(type(json_str))
print()
