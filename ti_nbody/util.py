from enum import Enum


class Method(Enum):
    Native = 1
    QuadTree = 2


def write_to_file(s):
    f = open("__created__.py", 'w')
    f.write(s)
    f.close()
