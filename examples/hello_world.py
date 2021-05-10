import pdb

import taichi as ti

from ti_nbody import n_body
from ti_nbody.init_functions import circle

ti.init(arch=ti.cpu)

DIM = 2
DT = 1e-5
NUM_MAX_PARTICLE = 32768  # 2^15
SHAPE_FACTOR = 1
particle_pos = ti.Vector.field(n=DIM, dtype=ti.f32)
particle_vel = ti.Vector.field(n=DIM, dtype=ti.f32)
particle_mass = ti.field(dtype=ti.f32)
particle_table = ti.root.dense(indices=ti.i, dimensions=NUM_MAX_PARTICLE)
particle_table.place(particle_pos).place(particle_vel).place(particle_mass)
num_particles = ti.field(dtype=ti.i32, shape=())


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

    # for i in range(10000):

    kernel(1)
    print(particle_pos.to_numpy())

    # while gui.running:
    #     gui.circles(particle_pos.to_numpy(), radius=2, color=0xfbfcbf)
    # gui.show()
    # kernel(1)
