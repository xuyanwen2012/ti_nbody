import pdb
import taichi as ti
from .util import *


def n_body(init_func, update_func, method=Method.Native):
    # pdb.set_trace()
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
import os.path
import importlib.util

cwd = os.getcwd()
path = os.path.join(cwd, 'examples\hello_world.py')
spec = importlib.util.spec_from_file_location('client', path)
imported = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imported)
print(imported.__dir__())

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

    if method == Method.Native:
        path = write_to_file(raw_kernel_str)
        print(path)

    generated_lib = import_from_site_packages('_created.py')

    generated_lib.circle(2 ** 10)

    return lambda _: generated_lib.substep()
