import pdb

import taichi as ti
import taichi.lang

ti.init()


@ti.kernel
def test_kernel_3() -> ti.f32:
    return ti.random()


if __name__ == '__main__':

    k_str = '''
def test_kernel() -> ti.f32:
    return ti.random()
    '''
    exec(k_str)

    # k = taichi.lang.kernel(test_kernel_3)
    k = taichi.lang.kernel(test_kernel)

    # pdb.set_trace()
    print(k())
