import yaml
from easydict import EasyDict as ED


def load_config(config_path="./config/config.yml"):
    """
    Function to load configuration file from provided yml File.
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        config = ED(yaml.load(file, Loader=yaml.SafeLoader))
    return config