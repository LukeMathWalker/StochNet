import os

def get_python_2_env():
    return '/home/' + os.environ['USER'] + '/anaconda3/envs/py2'


def get_python_3_env():
    return '/home/' + os.environ['USER'] + '/anaconda3'
