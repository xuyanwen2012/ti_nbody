import taichi as ti

from ti_nbody import n_body
from ti_nbody.init_functions import circle


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
    init = circle
    update = custom_gravity_func
    kernel = n_body(init, update)

    # Renderer related
    RES = (640, 480)
    gui = ti.GUI('N-body Star', res=RES)

    while gui.running:
        gui.circles(kernel.particle_pos.to_numpy(), radius=2, color=0xfbfcbf)
        gui.show()
        kernel.substep()
