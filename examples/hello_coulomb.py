import taichi as ti

from ti_nbody import n_body, Method


@ti.func
def custom_gravity_func(distance):
    l = distance.norm() + 1e-3
    return distance / (l ** 2)


@ti.kernel
def custom_init_func(num_p: ti.i32):
    for _ in range(num_p):
        particle_id = alloc_particle()
        particle_mass[particle_id] = 1.0 * ti.random() - 0.5
        particle_pos[particle_id] = ti.Vector([ti.random(), ti.random()])


if __name__ == '__main__':
    # Pick your ingredient for ti_nbody here, that's all it is
    init = (1024, custom_init_func)
    update = custom_gravity_func
    (kernel, gen_lib) = n_body(init, update, Method.Native)

    # GUI Renderer related
    RES = (640, 480)
    gui = ti.GUI('N-body Star', res=RES)

    while gui.running:
        gui.circles(gen_lib.particle_pos.to_numpy(), radius=2, color=0xfbfcbf)
        gui.show()
        kernel()
