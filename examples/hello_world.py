import taichi as ti
from ti_nbody import n_body


@ti.kernel
def custom_init_func(num_p: ti.i32):
    for _ in range(num_p):
        particle_id = alloc_particle()
        particle_mass[particle_id] = ti.random() * 1.4 + 0.1
        particle_pos[particle_id] = ti.Vector([ti.random(), ti.random()])


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
    init = custom_init_func
    update = custom_gravity_func
    kernel = n_body(init, update)

    # Renderer related
    RES = (640, 480)
    gui = ti.GUI('N-body Star', res=RES)

    # while gui.running:
    #     # gui.circles(particle_pos.to_numpy(), radius=2, color=0xfbfcbf)
    #     gui.show()
