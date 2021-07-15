import taichi as ti
import math

ti.init(arch=ti.cpu)

kShapeFactor = 1
kMaxParticles = 8192
kMaxDepth = kMaxParticles * 1
kMaxNodes = kMaxParticles * 4
kDim = 2

dt = 0.00005
LEAF = -1
TREE = -2

particle_mass = ti.field(ti.f32)
particle_pos = ti.Vector.field(kDim, ti.f32)
particle_vel = ti.Vector.field(kDim, ti.f32)
particle_table = ti.root.dense(ti.i, kMaxParticles)
particle_table.place(particle_pos).place(particle_vel).place(particle_mass)
particle_table_len = ti.field(ti.i32, ())


@ti.func
def alloc_particle():
    ret = ti.atomic_add(particle_table_len[None], 1)
    assert ret < kMaxParticles
    particle_mass[ret] = 0
    particle_pos[ret] = particle_pos[0] * 0
    particle_vel[ret] = particle_pos[0] * 0
    return ret


@ti.func
def reflect_boundary(pos,
                     vel,
                     pmin=0,
                     pmax=1,
                     rebound=1,
                     rebound_perpendicular=1):
    """
    Reflects particle velocity from a rectangular boundary (if collides).
    `boundaryReflect` takes particle position, velocity and other parameters.
    """
    cond = pos < pmin and vel < 0 or pos > pmax and vel > 0
    for j in ti.static(range(pos.n)):
        if cond[j]:
            vel[j] *= -rebound
            for k in ti.static(range(pos.n)):
                if k != j:
                    vel[k] *= rebound_perpendicular
    return vel
