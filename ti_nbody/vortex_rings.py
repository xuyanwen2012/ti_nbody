import uuid
from .util import *


def vortex_rings(init_func, update_func, method=Method.Native):
    particle_decl = '''eps = 0.01
dt = 0.1
n_vortex = 4
n_tracer = 200000
pos = ti.Vector.field(2, ti.f32, shape=n_vortex)
new_pos = ti.Vector.field(2, ti.f32, shape=n_vortex)
vort = ti.field(ti.f32, shape=n_vortex)
tracer = ti.Vector.field(2, ti.f32, shape=n_tracer)
'''

    # -------------------- N Squared --------------------------------------
    raw_str = '''@ti.func
def compute_u_full(p):
    u = ti.Vector([0.0, 0.0])
    for i in range(n_vortex):
        u += %s(p, pos[i], vort[i])
    return u
    
@ti.kernel
def integrate_vortex():
    for i in range(n_vortex):
        v = ti.Vector([0.0, 0.0])
        for j in range(n_vortex):
            if i != j:
                v += %s(pos[i], pos[j], vort[j])
        new_pos[i] = pos[i] + dt * v

    for i in range(n_vortex):
        pos[i] = new_pos[i]
''' % (update_func.__name__, update_func.__name__)

    raw_kernel_str = '''import taichi as ti
import numpy as np
import math
ti.init(arch=ti.gpu)
%s
%s
%s
%s
@ti.kernel
def advect():
    for i in range(n_tracer):
        # Ralston's third-order method
        p = tracer[i]
        v1 = compute_u_full(p)
        v2 = compute_u_full(p + v1 * dt * 0.5)
        v3 = compute_u_full(p + v2 * dt * 0.75)
        tracer[i] += (2 / 9 * v1 + 1 / 3 * v2 + 4 / 9 * v3) * dt

pos[0] = [0, 1]
pos[1] = [0, -1]
pos[2] = [0, 0.3]
pos[3] = [0, -0.3]
vort[0] = 1
vort[1] = -1
vort[2] = 1
vort[3] = -1
''' % (particle_decl,
       ti_func_to_string(init_func),
       ti_func_to_string(update_func),
       raw_str)

    # -------------------- generation -----------------------------------

    generated_name = "_created" + uuid.uuid1().hex + ".py"
    if method == Method.Native:
        write_to_file(generated_name, raw_kernel_str)
    else:
        pass

    generated_lib = import_from_site_packages(generated_name)
    # n_tracer = 200000
    generated_lib.init_tracers(200000)

    if method == Method.Native:
        def lam():
            # substeps
            for i in range(4):
                generated_lib.advect()
                generated_lib.integrate_vortex()

        return lam, generated_lib.tracer
    else:
        pass
