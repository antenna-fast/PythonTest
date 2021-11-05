import os
import json
import yaml
import pickle
import argparse


def mkdir_if_no_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_json_file(file_path):
    with open(file_path, 'r') as f:
        res = json.loads(f.read())
    return res


def dump_result_to_json(res, json_path):
    with open(json_path, 'w') as f:
        json.dump(res, f, indent=4)


def load_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        res = pickle.load(f)
    return res


def dump_result_to_pkl(res, pkl_path):
    with open(pkl_path, 'wb') as f:
        pickle.dump(res, f)


class YamlInputAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if os.path.exists(values):
            try:
                with open(values) as fp:
                    raw = fp.read()
            except Exception:
                raise argparse.ArgumentError(self, 'invalid yaml file')
        else:
            raw = values
        try:
            v = yaml.safe_load(raw)
            if not isinstance(v, dict):
                raise argparse.ArgumentError(
                    self, 'input file is not a dictionary'
                )
            v = self.parse_dict(v)

            setattr(namespace, self.dest, v)
            setattr(namespace, 'conf_file', values)
        except ValueError:
            raise argparse.ArgumentError(self, 'invalid yaml content')
    
    def parse_dict(self, v):
        args = argparse.Namespace()
        for _key, _value in v.items():
            if not isinstance(_value, dict):
                args.__setattr__(_key, _value)
            else:
                _args = self.parse_dict(_value)
                args.__setattr__(_key, _args)
        return args

