import taichi as ti
from ti_nbody import n_body, init_functions, Method


@ti.func
def custom_gravity_func(distance):
    l2 = distance.norm_sqr() + 1e-3
    return distance * (l2 ** ((-3) / 2))


if __name__ == '__main__':
    # Pick your ingredient for ti_nbody here
    init = init_functions.circle
    update = custom_gravity_func
    (kernel, particle_pos) = n_body(init, update, Method.QuadTree)

    # Renderer related
    RES = (640, 480)
    gui = ti.GUI('N-body Star', res=RES)

    while gui.running:
        gui.circles(particle_pos.to_numpy(), radius=2, color=0xfbfcbf)
        gui.show()
        kernel()
