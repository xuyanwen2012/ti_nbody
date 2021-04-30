import taichi as ti
import taichi.lang

ti.init()


def write_to_file(s):
    f = open("created.py", 'w')
    f.write(s)
    f.close()


if __name__ == '__main__':
    k_str = '''
import taichi as ti

@ti.kernel
def test_kernel() -> ti.f32:
    return ti.random()
    '''
    write_to_file(k_str)
    import created

    test_kernel = taichi.lang.kernel(created.test_kernel)
    print(test_kernel())
