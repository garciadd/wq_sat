import os
import yaml

homedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_base_dir():
    return homedir

def get_data_path():
    return os.path.join(os.path.dirname(homedir), 'data')

def load_credentials():
    if not os.path.isfile('credentials.yaml'):
        raise BadRequest('You must create a credentials.yaml file to store your {} user/pass'.format(name))
        
    with open('credentials.yaml', 'r') as f:
        return yaml.safe_load(f)
    