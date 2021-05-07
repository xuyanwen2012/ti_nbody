import inspect
import os.path
from enum import Enum


class Method(Enum):
    Native = 1
    QuadTree = 2


def write_to_file(s):
    path = os.path.join(os.path.dirname(__file__), "__created__.py")
    f = open(path, 'w')
    f.write(s)
    f.close()
    return path


def ti_func_to_string(func):
    lines = inspect.getsource(func)
    return lines
