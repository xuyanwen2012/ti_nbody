import taichi as ti
from .util import *

# ti.init(arch=ti.cpu)
#
# DT = 1e-5
# DIM = 2
# NUM_MAX_PARTICLE = 32768  # 2^15
# SHAPE_FACTOR = 1
# particle_pos = ti.Vector.field(n=DIM, dtype=ti.f32)
# particle_vel = ti.Vector.field(n=DIM, dtype=ti.f32)
# particle_mass = ti.field(dtype=ti.f32)
# particle_table = ti.root.dense(indices=ti.i, dimensions=NUM_MAX_PARTICLE)
# particle_table.place(particle_pos).place(particle_vel).place(particle_mass)
# num_particles = ti.field(dtype=ti.i32, shape=())


def n_body(init_func, update_func, method=Method.Native):
    raw_str = '''
@ti.func
def get_raw_gravity_at(pos):
    acc = particle_pos[0] * 0
    for i in range(num_particles[None]):
        acc += particle_mass[i] * %s(particle_pos[i] - pos)
    return acc
    
@ti.kernel
def substep():
    for i in range(num_particles[None]):
        acceleration = get_raw_gravity_at(particle_pos[i])
        particle_vel[i] += acceleration * DT

    for i in range(num_particles[None]):
        particle_pos[i] += particle_vel[i] * DT
''' % update_func.__name__

    raw_kernel_str = '''
import taichi as ti
import math

ti.init(arch=ti.cpu)

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

@ti.func
def alloc_particle():
    ret = ti.atomic_add(num_particles[None], 1)
    assert ret < NUM_MAX_PARTICLE
    particle_mass[ret] = 0
    particle_pos[ret] = particle_pos[0] * 0
    particle_vel[ret] = particle_pos[0] * 0
    return ret
    
%s
%s
%s
''' % (ti_func_to_string(init_func), ti_func_to_string(update_func), raw_str)

    generated_name = "_created.py"
    if method == Method.Native:
        path = write_to_file(generated_name, raw_kernel_str)
        print(path)

    generated_lib = import_from_site_packages(generated_name)

    generated_lib.circle(2 ** 10)
    print(generated_lib)

    # return lambda _: generated_lib.substep()
    return generated_lib
