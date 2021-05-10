import inspect
import os.path
import importlib.util
from enum import Enum


class Method(Enum):
    Native = 1
    QuadTree = 2


def import_from_site_packages(filename):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(dir_path, filename)
    return import_from('ti_nbody', path)


def import_from(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name,
                                                  file_path)
    created = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(created)
    return created


def write_to_file(s):
    path = os.path.join(os.path.dirname(__file__), "_created.py")
    f = open(path, 'w')
    f.write(s)
    f.close()
    return path


def ti_func_to_string(func):
    lines = inspect.getsource(func)
    return lines
