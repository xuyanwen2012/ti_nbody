import taichi as ti

from ti_nbody import n_body, Method
from ti_nbody.init_functions import uniform


@ti.func
def custom_gravity_func(distance):
    l2 = distance.norm_sqr() + 1e-3
    return distance * (l2 ** ((-3) / 2))


@ti.kernel
def custom_init_func(num_p: ti.i32):
    for _ in range(num_p):
        particle_id = alloc_particle()
        particle_mass[particle_id] = 1.5 * ti.random()

        # rand_disk_2d inlined
        x = 2 * ti.random() - 1
        y = 2 * ti.random() - 1
        while x * x + y * y > 1:
            x = 2 * ti.random() - 1
            y = 2 * ti.random() - 1
        rand_disk_2d = ti.Vector([x, y])

        particle_pos[particle_id] = ti.Vector([0.5, 0.5])

        # velocity = (particle_pos[particle_id] - 0.5) * angular_velocity * 250
        # particle_vel[particle_id] = ti.Vector([-velocity.y, velocity.x])


if __name__ == '__main__':
    # Pick your ingredient for ti_nbody here, that's all it is
    init = uniform
    update = custom_gravity_func
    (kernel, gen_lib) = n_body(init, update, Method.QuadTree)

    # GUI Renderer related
    RES = (640, 480)
    gui = ti.GUI('N-body Star', res=RES)

    while gui.running:
        gui.circles(gen_lib.particle_pos.to_numpy(), radius=2, color=0xfbfcbf)
        gui.show()
        kernel()
