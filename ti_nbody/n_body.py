import taichi as ti

from .util import Method, write_to_file


def n_body(init_func, update_func, method=Method.Native):
    declare_tables_str = '''
    DT = 1e-5
    DIM = 2
    NUM_MAX_PARTICLE = 32768  # 2^15
    SHAPE_FACTOR = 1

    particle_pos = ti.Vector.field(n=DIM, dtype=ti.f32)
    particle_vel = ti.Vector.field(n=DIM, dtype=ti.f32)
    particle_mass = ti.field(dtype=ti.f32)
    particle_table = ti.root.dense(indices=ti.i, dimensions=NUM_MAX_PARTICLE)
    particle_table.place(particle_pos).place(particle_vel).place(particle_mass)
    num_particles = ti.field(dtype=ti.i32, shape=())

    '''

    # @ti.func
    # def alloc_particle():
    #     ret = ti.atomic_add(num_particles[None], 1)
    #     assert ret < NUM_MAX_PARTICLE
    #     particle_mass[ret] = 0
    #     particle_pos[ret] = particle_pos[0] * 0
    #     particle_vel[ret] = particle_pos[0] * 0
    #     return ret
    #
    # @ti.func
    # def get_raw_gravity_at(pos):
    #     acc = particle_pos[0] * 0
    #     for i in range(num_particles[None]):
    #         acc += particle_mass[i] * % s(particle_pos[i] - pos)
    #     return acc

    # if method == Method.Native:
    #     write_to_file(raw_kernel_str)
    #
    # import __created__ as created

    # created.initialize(2 ** 10)
    #
    # return lambda _: created.substep()
    return 1