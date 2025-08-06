import yaml
from easydict import EasyDict as edict

def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return edict(cfg)
