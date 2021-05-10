import taichi as ti
import math


@ti.kernel
def circle(num_p: ti.i32):
    for _ in range(num_p):
        particle_id = alloc_particle()
        particle_mass[particle_id] = ti.random() * 1.4 + 0.1
        a = ti.random() * math.tau
        r = ti.sqrt(ti.random()) * 0.3
        particle_pos[particle_id] = 0.5 + ti.Vector(
            [ti.cos(a), ti.sin(a)]) * r
        # print(particle_pos[particle_id])


@ti.kernel
def uniform(num_p: ti.i32):
    for _ in range(num_p):
        particle_id = alloc_particle()
        particle_mass[particle_id] = ti.random() * 1.4 + 0.1
        particle_pos[particle_id] = ti.Vector([ti.random(), ti.random()])
