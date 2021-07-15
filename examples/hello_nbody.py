import taichi as ti

from ti_nbody import n_body, Method


# from ti_nbody.init_functions import circle


@ti.func
def custom_gravity_func(distance):
    l2 = distance.norm_sqr() + 1e-3
    return distance * (l2 ** ((-3) / 2))


@ti.kernel
def custom_init_func(num_p: ti.i32):
    for _ in range(num_p):
        particle_id = alloc_particle()
        particle_mass[particle_id] = ti.random() * 1.4 + 0.1
        particle_pos[particle_id] = ti.Vector([ti.random(), ti.random()])


if __name__ == '__main__':
    # Pick your ingredient for ti_nbody here, that's all it is
    init = custom_init_func
    update = custom_gravity_func
    (kernel, particle_pos) = n_body(init, update, Method.Native)

    # GUI Renderer related
    RES = (640, 480)
    gui = ti.GUI('N-body Star', res=RES)

    while gui.running:
        gui.circles(particle_pos.to_numpy(), radius=2, color=0xfbfcbf)
        gui.show()
        kernel()
