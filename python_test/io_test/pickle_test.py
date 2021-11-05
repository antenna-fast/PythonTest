import pickle

#  保存字典
dic = {'name', 123}


def save_obj(obj, path_name):
    with open(path_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path_name):
    with open(path_name, 'rb') as f:
        return pickle.load(f)


save_obj(dic, '123/tt.pkl')
dic1 = load_obj('123/tt.pkl')
print(dic1)
