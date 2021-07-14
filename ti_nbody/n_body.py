import uuid
from .util import *
import time


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def timed_kernel(init_func, update_func, method=Method.Native):
    t0 = time.time()
    n_body(init_func, update_func, method)
    t1 = time.time()
    print(f'Time: {t1 - t0}')


def n_body(init_func, update_func, method=Method.Native):
    particle_decl = '''import taichi as ti
import math
ti.init()
if not hasattr(ti, 'jkl'):
    ti.jkl = ti.indices(1, 2, 3)
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
    # -------------------- N Squared --------------------------------------
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

    raw_kernel_str = '''import taichi as ti
import math
%s
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
''' % (particle_decl,
       ti_func_to_string(init_func),
       ti_func_to_string(update_func),
       raw_str)

    # -------------------- Quad Tree method -----------------------------------

    tree_str = '''
@ti.func
def get_gravity_at(position):
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
            acc += particle_mass[particle_id] * %s(distance)

        else:  # TREE or LEAF
            for which in ti.grouped(ti.ndrange(*([2] * DIM))):
                child = node_children[parent, which]
                if child == LEAF:
                    continue
                node_center = node_centroid_pos[child] / node_mass[child]
                distance = node_center - position
                if distance.norm_sqr() > \
                        SHAPE_FACTOR ** 2 * parent_geo_size ** 2:
                    acc += node_mass[child] * %s(distance)
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
    while particle_id < num_particles[None]:
        acceleration = get_gravity_at(particle_pos[particle_id])
        particle_vel[particle_id] += acceleration * DT
        # well... seems our tree inserter will break if particle out-of-bound:
        particle_vel[particle_id] = boundReflect(
            particle_pos[particle_id],
            particle_vel[particle_id],
            0, 1)
        particle_id = particle_id + 1

    for i in range(num_particles[None]):
        particle_pos[i] += particle_vel[i] * DT
''' % (update_func.__name__, update_func.__name__)

    tree_kernel_str = '''
%s
T_MAX_DEPTH = 8
T_MAX_NODES = 4 * T_MAX_DEPTH
LEAF = -1
TREE = -2
node_mass = ti.field(ti.f32)
node_centroid_pos = ti.Vector.field(DIM, ti.f32)
node_particle_id = ti.field(ti.i32)
node_children = ti.field(ti.i32)
node_table = ti.root.dense(ti.i, T_MAX_NODES)
node_table.place(node_particle_id, node_centroid_pos, node_mass)
node_table.dense(indices={2: ti.jk, 3: ti.jkl}[DIM], dimensions=2).place(node_children)
node_table_len = ti.field(dtype=ti.i32, shape=())
trash_particle_id = ti.field(ti.i32)
trash_base_parent = ti.field(ti.i32)
trash_base_geo_center = ti.Vector.field(DIM, ti.f32)
trash_base_geo_size = ti.field(ti.f32)
trash_table = ti.root.dense(ti.i, T_MAX_DEPTH)
trash_table.place(trash_particle_id)
trash_table.place(trash_base_parent, trash_base_geo_size)
trash_table.place(trash_base_geo_center)
trash_table_len = ti.field(ti.i32, ())

@ti.func
def alloc_particle():
    ret = ti.atomic_add(num_particles[None], 1)
    assert ret < NUM_MAX_PARTICLE
    particle_mass[ret] = 0
    particle_pos[ret] = particle_pos[0] * 0
    particle_vel[ret] = particle_pos[0] * 0
    return ret

@ti.func
def alloc_node():
    ret = ti.atomic_add(node_table_len[None], 1)
    assert ret < T_MAX_NODES

    node_mass[ret] = 0
    node_centroid_pos[ret] = particle_pos[0] * 0

    # indicate the 4 children to be LEAF as well
    node_particle_id[ret] = LEAF
    for which in ti.grouped(ti.ndrange(*([2] * DIM))):
        node_children[ret, which] = LEAF
    return ret

@ti.func
def alloc_trash():
    ret = ti.atomic_add(trash_table_len[None], 1)
    assert ret < T_MAX_DEPTH
    return ret

@ti.func
def alloc_a_node_for_particle(particle_id, parent, parent_geo_center,
                              parent_geo_size):
    position = particle_pos[particle_id]
    mass = particle_mass[particle_id]

    # (Making sure not to parallelize this loop)
    # Traversing down the tree to find a suitable location for the particle.
    depth = 0
    while depth < T_MAX_DEPTH:
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
            # Subtract pos/mass of the particle from the parent node
            already_pos = particle_pos[already_particle_id]
            already_mass = particle_mass[already_particle_id]
            node_centroid_pos[parent] -= already_pos * already_mass
            node_mass[parent] -= already_mass

        node_centroid_pos[parent] += position * mass
        node_mass[parent] += mass

        # Determine which quadrant (as 'child') this particle shout go into.
        which = abs(position > parent_geo_center)
        child = node_children[parent, which]
        if child == LEAF:
            child = alloc_node()
            node_children[parent, which] = child

        # the geo size of this level should be halved
        child_geo_size = parent_geo_size * 0.5
        child_geo_center = parent_geo_center + (which - 0.5) * child_geo_size

        parent_geo_center = child_geo_center
        parent_geo_size = child_geo_size
        parent = child

        depth = depth + 1

    # Note, parent here was used as like a 'current' in iterative
    node_particle_id[parent] = particle_id
    node_centroid_pos[parent] = position * mass
    node_mass[parent] = mass

@ti.kernel
def build_tree():
    """
    Once the 'particle table' is populated, we can construct a 'node table',
    which contains all the node information, and construct the child table as
    well.
    :return:
    """
    node_table_len[None] = 0
    trash_table_len[None] = 0
    alloc_node()

    # (Making sure not to parallelize this loop)
    # Foreach particle: register it to a node.
    particle_id = 0
    while particle_id < num_particles[None]:
        # Root as parent,
        # 0.5 (center) as the parent centroid position
        # 1.0 (whole) as the parent geo size
        alloc_a_node_for_particle(particle_id, 0,
                                  particle_pos[0] * 0 + 0.5,
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
%s
%s
%s

@ti.func
def boundReflect(pos, vel, pmin=0, pmax=1, gamma=1, gamma_perpendicular=1):
    cond = pos < pmin and vel < 0 or pos > pmax and vel > 0
    for j in ti.static(range(pos.n)):
        if cond[j]:
            vel[j] *= -gamma
            for k in ti.static(range(pos.n)):
                if k != j:
                    vel[k] *= gamma_perpendicular
    return vel

    ''' % (particle_decl,
           ti_func_to_string(init_func),
           ti_func_to_string(update_func),
           tree_str)

    # -------------------- generation -----------------------------------

    generated_name = "_created" + uuid.uuid1().hex + ".py"
    if method == Method.Native:
        write_to_file(generated_name, raw_kernel_str)
    elif method == Method.QuadTree:
        write_to_file(generated_name, tree_kernel_str)

    generated_lib = import_from_site_packages(generated_name)
    generated_lib.circle(2 ** 10)

    if method == Method.Native:
        def lam():
            generated_lib.substep()

        return lam, generated_lib.particle_pos
    elif method == Method.QuadTree:
        @static_vars(counter=0, total_time_build=0, total_time_substep=0)
        def lam():
            generated_lib.build_tree()
            generated_lib.substep()

        generated_lib.build_tree()
        generated_lib.substep()

        return lam, generated_lib.particle_pos
