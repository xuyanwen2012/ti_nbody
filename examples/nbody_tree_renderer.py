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


@ti.kernel
def init_func(num_p: ti.i32):
    for _ in range(num_p):
        particle_id = alloc_particle()
        particle_mass[particle_id] = ti.random() * 1.4 + 0.1
        particle_pos[particle_id] = ti.Vector([ti.random(), ti.random()])


@ti.func
def custom_gravity_func(distance):
    l2 = distance.norm_sqr() + 1e-3
    return distance * (l2 ** ((-3) / 2))


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
    node_weighted_pos[ret] = particle_pos[0] * 0
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
    position = particle_pos[particle_id]
    mass = particle_mass[particle_id]

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
            already_pos = particle_pos[already_particle_id]
            already_mass = particle_mass[already_particle_id]
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
    while particle_id < particle_table_len[None]:
        alloc_a_node_for_particle(particle_id, 0, particle_pos[0] * 0 + 0.5,
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
    acc = particle_pos[0] * 0

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
            distance = particle_pos[particle_id] - position
            acc += particle_mass[particle_id] * custom_gravity_func(distance)

        else:  # TREE or LEAF
            for which in ti.grouped(ti.ndrange(*([2] * kDim))):
                child = node_children[parent, which]
                if child == LEAF:
                    continue
                node_center = node_weighted_pos[child] / node_mass[child]
                distance = node_center - position
                if distance.norm_sqr() > kShapeFactor ** 2 * parent_geo_size ** 2:
                    acc += node_mass[child] * custom_gravity_func(distance)
                else:
                    new_trash_id = alloc_trash()
                    child_geo_size = parent_geo_size * 0.5
                    trash_base_parent[new_trash_id] = child
                    trash_base_geo_size[new_trash_id] = child_geo_size

        trash_id = trash_id + 1

    return acc


@ti.kernel
def substep():
    particle_id = 0
    while particle_id < particle_table_len[None]:
        acceleration = get_tree_gravity_at(particle_pos[particle_id])
        particle_vel[particle_id] += acceleration * dt
        # well... seems our tree inserter will break if a particle is out-of-bound:
        particle_vel[particle_id] = reflect_boundary(particle_pos[particle_id],
                                                     particle_vel[particle_id],
                                                     0, 1)
        particle_id = particle_id + 1
    for i in range(particle_table_len[None]):
        particle_pos[i] += particle_vel[i] * dt


def render_tree(gui,
                parent=0,
                parent_geo_center=ti.Vector([0.5, 0.5]),
                parent_geo_size=1.0):
    child_geo_size = parent_geo_size * 0.5
    if node_particle_id[parent] >= 0:
        tl = parent_geo_center - child_geo_size
        br = parent_geo_center + child_geo_size
        gui.rect(tl, br, radius=1, color=0xff0000)
    for which in map(ti.Vector, [[0, 0], [0, 1], [1, 0], [1, 1]]):
        child = node_children[(parent, which[0], which[1])]
        if child < 0:
            continue
        a = parent_geo_center + (which - 1) * child_geo_size
        b = parent_geo_center + which * child_geo_size
        child_geo_center = parent_geo_center + (which - 0.5) * child_geo_size
        gui.rect(a, b, radius=1, color=0xff0000)
        render_tree(gui, child, child_geo_center, child_geo_size)


if __name__ == '__main__':
    # GUI Renderer related
    gui = ti.GUI('N-body Star', res=(1280, 960))

    init_func(64)

    while gui.running:
        gui.circles(particle_pos.to_numpy(), radius=2, color=0xfbfcbf)

        build_tree()
        substep()

        render_tree(gui)

        gui.show()
