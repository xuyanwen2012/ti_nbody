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

# --------

a_particle_mass = ti.field(ti.f32)
a_particle_pos = ti.Vector.field(kDim, ti.f32)
a_particle_vel = ti.Vector.field(kDim, ti.f32)
a_particle_table = ti.root.dense(ti.i, kMaxParticles)
a_particle_table.place(a_particle_pos).place(a_particle_vel).place(
    a_particle_mass)
a_particle_table_len = ti.field(ti.i32, ())


@ti.func
def a_alloc_particle():
    ret = ti.atomic_add(a_particle_table_len[None], 1)
    assert ret < kMaxParticles
    a_particle_mass[ret] = 0
    a_particle_pos[ret] = a_particle_pos[0] * 0
    a_particle_vel[ret] = a_particle_pos[0] * 0
    return ret


@ti.kernel
def a_init_func(num_p: ti.i32):
    for _ in range(num_p):
        particle_id = a_alloc_particle()
        a_particle_mass[particle_id] = ti.random() * 1.4 + 0.1
        a_particle_pos[particle_id] = ti.Vector([ti.random(), ti.random()])


@ti.func
def get_raw_gravity_at(pos):
    acc = a_particle_pos[0] * 0
    for i in range(a_particle_table_len[None]):
        acc += a_particle_mass[i] * gravity_func(a_particle_pos[i] - pos)
    return acc


@ti.kernel
def a_substep():
    for i in range(a_particle_table_len[None]):
        acceleration = get_raw_gravity_at(a_particle_pos[i])
        a_particle_vel[i] += acceleration * dt
        # We dont have to do it here, but we want to make sure the results is
        # aligned with the quadtree approach.
        a_particle_vel[i] = reflect_boundary(a_particle_pos[i],
                                             a_particle_vel[i],
                                             0, 1)
    for i in range(a_particle_table_len[None]):
        a_particle_pos[i] += a_particle_vel[i] * dt


# --------

b_particle_mass = ti.field(ti.f32)
b_particle_pos = ti.Vector.field(kDim, ti.f32)
b_particle_vel = ti.Vector.field(kDim, ti.f32)
b_particle_table = ti.root.dense(ti.i, kMaxParticles)
b_particle_table.place(b_particle_pos).place(b_particle_vel).place(
    b_particle_mass)
b_particle_table_len = ti.field(ti.i32, ())


# --------

@ti.func
def gravity_func(distance):
    l2 = distance.norm_sqr() + 1e-3
    return distance * (l2 ** ((-3) / 2))


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


if __name__ == '__main__':
    RES = (640, 480)
    gui = ti.GUI('N-body Star', res=RES)

    a_init_func(8192)

    while gui.running:
        gui.circles(a_particle_pos.to_numpy(), radius=2, color=0xfbfcbf)
        gui.show()
        a_substep()
