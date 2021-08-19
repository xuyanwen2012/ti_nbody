import uuid
from .util import *


def n_body(init, update_func, method=Method.Native, threads=1, theta=1.0):
    (num, init_func) = init

    kernel_str = read_ti_files('common')
    kernel_str += f'\n\n{ti_func_to_string(init_func)}'.replace(
        init_func.__name__, 'init_func')

    # Replace params in the kernel string
    kernel_str = kernel_str.replace("PARTICLE_PARAM", str(num))
    kernel_str = kernel_str.replace("THETA_PARAM", str(theta))
    kernel_str = kernel_str.replace("NUM_THREADS_PARAM", str(threads))
    
    kernel_str += f'\n\n{ti_func_to_string(update_func)}\n\n'

    generated_name = "_created" + uuid.uuid1().hex + ".py"
    if method == Method.Native:
        kernel_str += read_ti_files('native')
    elif method == Method.QuadTree:
        kernel_str += read_ti_files('bh')

    kernel_str = kernel_str.replace('__GRAVITY_FUNC_NAME__',
                                    update_func.__name__)

    # print('========================================================')
    # print(kernel_str)
    # print('========================================================')

    write_to_file(generated_name, kernel_str)

    generated_lib = import_from_site_packages(generated_name)
    generated_lib.init_func(num)

    if method == Method.Native:
        def lam():
            generated_lib.substep()

        return lam, generated_lib
    elif method == Method.QuadTree:
        def lam():
            generated_lib.build_tree()
            generated_lib.substep()

        return lam, generated_lib
