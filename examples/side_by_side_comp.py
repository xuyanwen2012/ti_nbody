import numpy as np
import taichi as ti

from ti_nbody import n_body, init_functions, Method


@ti.func
def custom_gravity_func(distance):
    """
    Define which ever the equation used to compute gravity here
    :param distance: the distance between things.
    :return:
    """
    l2 = distance.norm_sqr() + 1e-3
    return distance * (l2 ** ((-3) / 2))


if __name__ == '__main__':
    # Pick your ingredient for ti_nbody here
    init = init_functions.circle
    update = custom_gravity_func

    (raw_kernel, raw_par_pos) = n_body(init, update, Method.Native)
    (tree_kernel, tree_par_pos) = n_body(init, update, Method.QuadTree)

    # Renderer related
    w = 640
    h = 480
    gui = ti.GUI('N-body Star', res=(w * 2, h))

    while gui.running:
        raw_pos = raw_par_pos.to_numpy() / 2
        tree_pos = raw_par_pos.to_numpy() / 2 + 0.5

        gui.circles(np.append(raw_pos, tree_pos), radius=2, color=0xfbfcbf)
        gui.show()

        raw_kernel()
        tree_kernel()
