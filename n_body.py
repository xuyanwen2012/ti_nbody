import taichi as ti
import math

ti.init(arch=ti.cpu)
if not hasattr(ti, 'jkl'):
    ti.jkl = ti.indices(1, 2, 3)

# Program related
RES = (640, 480)

# N-body related
DT = 1e-5
DIM = 2
NUM_MAX_PARTICLE = 32768  # 2^15
# NUM_MAX_PARTICLE = 8192  # 2^13
SHAPE_FACTOR = 1

# ----------------------- Raw ------------------------------------------------

# Using this table to store all the information (pos, vel, mass) of particles
# Currently using SoA memory model
raw_num_particles = ti.field(dtype=ti.i32, shape=())
raw_particle_pos = ti.Vector.field(n=DIM, dtype=ti.f32)
raw_particle_vel = ti.Vector.field(n=DIM, dtype=ti.f32)
raw_particle_mass = ti.field(dtype=ti.f32)
raw_particle_table = ti.root.dense(indices=ti.i, dimensions=NUM_MAX_PARTICLE)
raw_particle_table.place(raw_particle_pos) \
    .place(raw_particle_vel) \
    .place(raw_particle_mass)


@ti.func
def raw_alloc_particle():
    """
    Always use this function to obtain an new particle id to operate on.
    :return: The ID of the just allocated particle
    """
    ret = ti.atomic_add(raw_num_particles[None], 1)
    assert ret < NUM_MAX_PARTICLE
    raw_particle_mass[ret] = 0
    raw_particle_pos[ret] = raw_particle_pos[0] * 0
    raw_particle_vel[ret] = raw_particle_pos[0] * 0
    return ret


@ti.func
def raw_get_gravity_at(pos):
    acc = raw_particle_pos[0] * 0
    for i in range(raw_num_particles[None]):
        acc += raw_particle_mass[i] * gravity_func(raw_particle_pos[i] - pos)
    return acc


# The O(N^2) kernel algorithm
@ti.kernel
def raw_substep():
    for i in range(raw_num_particles[None]):
        acceleration = raw_get_gravity_at(raw_particle_pos[i])
        raw_particle_vel[i] += acceleration * DT

    for i in range(raw_num_particles[None]):
        raw_particle_pos[i] += raw_particle_vel[i] * DT


# ----------------------- Tree ------------------------------------------------

# ----------------------- Shared ------------------------------------------------

@ti.func
def gravity_func(distance):
    """
    Define which ever the equation used to compute gravity here
    :param distance: the distance between things.
    :return:
    """
    # --- The equation defined in the new n-body example
    # (self**2).sum()
    l2 = distance.norm_sqr() + 1e-3
    return distance * (l2 ** ((-3) / 2))


@ti.kernel
def initialize(num_p: ti.i32):
    """
    Randomly set the initial position of the particles to start with. Note
    set a value to 'num_particles[None]' taichi field to indicate.
    :return: None
    """
    for _ in range(num_p):
        particle_id = raw_alloc_particle()

        raw_particle_mass[particle_id] = ti.random() * 1.4 + 0.1

        a = ti.random() * math.tau
        r = ti.sqrt(ti.random()) * 0.3
        raw_particle_pos[particle_id] = 0.5 + ti.Vector(
            [ti.cos(a), ti.sin(a)]) * r


if __name__ == '__main__':
    import numpy as np
    import sys

    # get command line as input
    assert len(sys.argv) == 2
    exp = int(sys.argv[1])
    initialize(2 ** exp)

    for i in range(1000):
        raw_substep()
