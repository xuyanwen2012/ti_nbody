import taichi as ti
import math
import numpy as np

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


@ti.func
def b_alloc_particle():
    ret = ti.atomic_add(b_particle_table_len[None], 1)
    assert ret < kMaxParticles
    b_particle_mass[ret] = 0
    b_particle_pos[ret] = b_particle_pos[0] * 0
    b_particle_vel[ret] = b_particle_pos[0] * 0
    return ret


trash_particle_id = ti.field(ti.i32)
trash_base_parent = ti.field(ti.i32)
trash_base_geo_center = ti.Vector.field(kDim, ti.f32)
trash_base_geo_size = ti.field(ti.f32)
trash_table = ti.root.dense(ti.i, kMaxDepth)
trash_table.place(trash_particle_id)
trash_table.place(trash_base_parent, trash_base_geo_size)
trash_table.place(trash_base_geo_center)
trash_table_len = ti.field(ti.i32, ())

node_mass = ti.field(ti.f32)
node_weighted_pos = ti.Vector.field(kDim, ti.f32)
node_particle_id = ti.field(ti.i32)
node_children = ti.field(ti.i32)
node_table = ti.root.dense(ti.i, kMaxNodes)
node_table.place(node_mass, node_particle_id, node_weighted_pos)
node_table.dense(ti.indices(*list(range(1, 1 + kDim))), 2).place(node_children)
node_table_len = ti.field(ti.i32, ())


@ti.func
def alloc_node():
    ret = ti.atomic_add(node_table_len[None], 1)
    assert ret < kMaxNodes
    node_mass[ret] = 0
    node_weighted_pos[ret] = b_particle_pos[0] * 0
    node_particle_id[ret] = LEAF
    for which in ti.grouped(ti.ndrange(*([2] * kDim))):
        node_children[ret, which] = LEAF
    return ret


@ti.func
def alloc_trash():
    ret = ti.atomic_add(trash_table_len[None], 1)
    assert ret < kMaxDepth
    return ret


@ti.func
def alloc_a_node_for_particle(particle_id, parent, parent_geo_center,
                              parent_geo_size):
    position = b_particle_pos[particle_id]
    mass = b_particle_mass[particle_id]

    depth = 0
    while depth < kMaxDepth:
        already_particle_id = node_particle_id[parent]
        if already_particle_id == LEAF:
            break
        if already_particle_id != TREE:
            node_particle_id[parent] = TREE
            trash_id = alloc_trash()
            trash_particle_id[trash_id] = already_particle_id
            trash_base_parent[trash_id] = parent
            trash_base_geo_center[trash_id] = parent_geo_center
            trash_base_geo_size[trash_id] = parent_geo_size
            already_pos = b_particle_pos[already_particle_id]
            already_mass = b_particle_mass[already_particle_id]
            node_weighted_pos[parent] -= already_pos * already_mass
            node_mass[parent] -= already_mass

        node_weighted_pos[parent] += position * mass
        node_mass[parent] += mass

        which = abs(position > parent_geo_center)
        child = node_children[parent, which]
        if child == LEAF:
            child = alloc_node()
            node_children[parent, which] = child
        child_geo_size = parent_geo_size * 0.5
        child_geo_center = parent_geo_center + (which - 0.5) * child_geo_size

        parent_geo_center = child_geo_center
        parent_geo_size = child_geo_size
        parent = child

        depth = depth + 1

    node_particle_id[parent] = particle_id
    node_weighted_pos[parent] = position * mass
    node_mass[parent] = mass


@ti.kernel
def build_tree():
    node_table_len[None] = 0
    trash_table_len[None] = 0
    alloc_node()

    particle_id = 0
    while particle_id < b_particle_table_len[None]:
        alloc_a_node_for_particle(particle_id, 0, b_particle_pos[0] * 0 + 0.5,
                                  1.0)

        trash_id = 0
        while trash_id < trash_table_len[None]:
            alloc_a_node_for_particle(trash_particle_id[trash_id],
                                      trash_base_parent[trash_id],
                                      trash_base_geo_center[trash_id],
                                      trash_base_geo_size[trash_id])
            trash_id = trash_id + 1

        trash_table_len[None] = 0
        particle_id = particle_id + 1


@ti.func
def get_tree_gravity_at(position):
    acc = b_particle_pos[0] * 0

    trash_table_len[None] = 0
    trash_id = alloc_trash()
    assert trash_id == 0
    trash_base_parent[trash_id] = 0
    trash_base_geo_size[trash_id] = 1.0

    trash_id = 0
    while trash_id < trash_table_len[None]:
        parent = trash_base_parent[trash_id]
        parent_geo_size = trash_base_geo_size[trash_id]

        particle_id = node_particle_id[parent]
        if particle_id >= 0:
            distance = b_particle_pos[particle_id] - position
            acc += b_particle_mass[particle_id] * gravity_func(distance)

        else:  # TREE or LEAF
            for which in ti.grouped(ti.ndrange(*([2] * kDim))):
                child = node_children[parent, which]
                if child == LEAF:
                    continue
                node_center = node_weighted_pos[child] / node_mass[child]
                distance = node_center - position
                if distance.norm_sqr() > kShapeFactor ** 2 * parent_geo_size ** 2:
                    acc += node_mass[child] * gravity_func(distance)
                else:
                    new_trash_id = alloc_trash()
                    child_geo_size = parent_geo_size * 0.5
                    trash_base_parent[new_trash_id] = child
                    trash_base_geo_size[new_trash_id] = child_geo_size

        trash_id = trash_id + 1

    return acc


@ti.kernel
def b_substep():
    particle_id = 0
    while particle_id < b_particle_table_len[None]:
        acceleration = get_tree_gravity_at(b_particle_pos[particle_id])
        b_particle_vel[particle_id] += acceleration * dt
        b_particle_vel[particle_id] = reflect_boundary(
            b_particle_pos[particle_id],
            b_particle_vel[particle_id],
            0, 1)
        particle_id = particle_id + 1
    for i in range(b_particle_table_len[None]):
        b_particle_pos[i] += b_particle_vel[i] * dt


# --------

@ti.kernel
def both_init_func(num_p: ti.i32):
    for _ in range(num_p):
        mass = ti.random() * 1.4 + 0.1
        rnd_pos = ti.Vector([ti.random(), ti.random()])

        a_particle_id = a_alloc_particle()
        b_particle_id = b_alloc_particle()

        a_particle_mass[a_particle_id] = mass
        a_particle_pos[a_particle_id] = rnd_pos.copy()

        b_particle_mass[b_particle_id] = mass
        b_particle_pos[b_particle_id] = rnd_pos.copy()


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
    RES = (640 * 2, 480)
    gui = ti.GUI('N-body Star', res=RES)

    both_init_func(8192)

    while gui.running:
        lhs = a_particle_pos.to_numpy()  # truth
        rhs = b_particle_pos.to_numpy()

        diffs = (rhs - lhs).sum()
        print(diffs)

        lhs /= (2, 1)
        rhs /= (2, 1)
        rhs += (0.5, 0)
        result = np.concatenate((lhs, rhs), axis=0)

        gui.circles(result, radius=1.5, color=0xfbfcbf)
        gui.show()

        a_substep()

        build_tree()
        b_substep()

        # do a simple comparison
