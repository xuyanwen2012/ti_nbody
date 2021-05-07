import pdb

import taichi as ti
from enum import Enum


class Method(Enum):
    Native = 1
    QuadTree = 2


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

    # ---------- Raw -----------------
    pdb.set_trace()
    raw_kernel_str = '''
import taichi as ti
import math

ti.init()

@ti.func
def alloc_particle():
    ret = ti.atomic_add(num_particles[None], 1)
    assert ret < NUM_MAX_PARTICLE
    particle_mass[ret] = 0
    particle_pos[ret] = particle_pos[0] * 0
    particle_vel[ret] = particle_pos[0] * 0
    return ret


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


%s
''' % (update_func.__name__, declare_tables_str + init_func)

    #     # ---------- Quad Tree -----------------
    #     quadtree_str = '''
    # import taichi as ti
    # import math
    #
    # ti.init()
    # ti.init(arch=ti.cpu)
    # if not hasattr(ti, 'jkl'):
    #     ti.jkl = ti.indices(1, 2, 3)
    #
    # %s
    #
    # @ti.func
    # def alloc_particle():
    #     ret = ti.atomic_add(num_particles[None], 1)
    #     assert ret < NUM_MAX_PARTICLE
    #     particle_mass[ret] = 0
    #     particle_pos[ret] = particle_pos[0] * 0
    #     particle_vel[ret] = particle_pos[0] * 0
    #     return ret
    #
    #
    # # Quadtree related
    # T_MAX_DEPTH = 1 * NUM_MAX_PARTICLE
    # T_MAX_NODES = 4 * T_MAX_DEPTH
    # LEAF = -1
    # TREE = -2
    #
    # # Each node contains information about the node mass, the centroid position,
    # # and the particle which it contains in ID
    # node_mass = ti.field(ti.f32)
    # node_centroid_pos = ti.Vector.field(DIM, ti.f32)
    # node_particle_id = ti.field(ti.i32)
    # node_children = ti.field(ti.i32)
    #
    # node_table = ti.root.dense(ti.i, T_MAX_NODES)
    # # node_table.place(node_mass, node_particle_id, node_centroid_pos)
    # node_table.place(node_particle_id, node_centroid_pos, node_mass)  # AoS here
    # node_table.dense(indices={2: ti.jk, 3: ti.jkl}[DIM], dimensions=2).place(
    #     node_children)  # ????
    # node_table_len = ti.field(dtype=ti.i32, shape=())
    #
    # # Also a trash table
    # trash_particle_id = ti.field(ti.i32)
    # trash_base_parent = ti.field(ti.i32)
    # trash_base_geo_center = ti.Vector.field(DIM, ti.f32)
    # trash_base_geo_size = ti.field(ti.f32)
    # trash_table = ti.root.dense(ti.i, T_MAX_DEPTH)
    # trash_table.place(trash_particle_id)
    # trash_table.place(trash_base_parent, trash_base_geo_size)
    # trash_table.place(trash_base_geo_center)
    # trash_table_len = ti.field(ti.i32, ())
    #
    #
    # @ti.func
    # def alloc_node():
    #     """
    #     Increment the current node table length, clear and set initial values
    #     (mass/centroid) to zeros of the allocated. The children information is
    #     stored in the 'node_children' table.
    #     :return: the ID of the just allocated node
    #     """
    #     ret = ti.atomic_add(node_table_len[None], 1)
    #     assert ret < T_MAX_NODES
    #
    #     node_mass[ret] = 0
    #     node_centroid_pos[ret] = particle_pos[0] * 0
    #
    #     # indicate the 4 children to be LEAF as well
    #     node_particle_id[ret] = LEAF
    #     for which in ti.grouped(ti.ndrange(*([2] * DIM))):
    #         node_children[ret, which] = LEAF
    #     return ret
    #
    #
    # @ti.func
    # def alloc_trash():
    #     ret = ti.atomic_add(trash_table_len[None], 1)
    #     assert ret < T_MAX_DEPTH
    #     return ret
    #
    #
    # @ti.func
    # def alloc_a_node_for_particle(particle_id, parent, parent_geo_center,
    #                               parent_geo_size):
    #     """
    #
    #     :param particle_id: The particle to be registered
    #     :param parent:
    #     :param parent_geo_center:
    #     :param parent_geo_size:
    #     """
    #     position = particle_pos[particle_id]
    #     mass = particle_mass[particle_id]
    #
    #     # (Making sure not to parallelize this loop)
    #     # Traversing down the tree to find a suitable location for the particle.
    #     depth = 0
    #     while depth < T_MAX_DEPTH:
    #         already_particle_id = node_particle_id[parent]
    #         if already_particle_id == LEAF:
    #             break
    #         if already_particle_id != TREE:
    #             node_particle_id[parent] = TREE
    #             trash_id = alloc_trash()
    #             trash_particle_id[trash_id] = already_particle_id
    #             trash_base_parent[trash_id] = parent
    #             trash_base_geo_center[trash_id] = parent_geo_center
    #             trash_base_geo_size[trash_id] = parent_geo_size
    #             # Subtract pos/mass of the particle from the parent node
    #             already_pos = particle_pos[already_particle_id]
    #             already_mass = particle_mass[already_particle_id]
    #             node_centroid_pos[parent] -= already_pos * already_mass
    #             node_mass[parent] -= already_mass
    #
    #         node_centroid_pos[parent] += position * mass
    #         node_mass[parent] += mass
    #
    #         # Determine which quadrant (as 'child') this particle shout go into.
    #         which = abs(position > parent_geo_center)
    #         child = node_children[parent, which]
    #         if child == LEAF:
    #             child = alloc_node()
    #             node_children[parent, which] = child
    #
    #         # the geo size of this level should be halved
    #         child_geo_size = parent_geo_size * 0.5
    #         child_geo_center = parent_geo_center + (which - 0.5) * child_geo_size
    #
    #         parent_geo_center = child_geo_center
    #         parent_geo_size = child_geo_size
    #         parent = child
    #
    #         depth = depth + 1
    #
    #     # Note, parent here was used as like a 'current' in iterative
    #     node_particle_id[parent] = particle_id
    #     node_centroid_pos[parent] = position * mass
    #     node_mass[parent] = mass
    #
    #
    # @ti.kernel
    # def build_tree():
    #     """
    #     Once the 'particle table' is populated, we can construct a 'node table',
    #     which contains all the node information, and construct the child table as
    #     well.
    #     :return:
    #     """
    #     node_table_len[None] = 0
    #     trash_table_len[None] = 0
    #     alloc_node()
    #
    #     # (Making sure not to parallelize this loop)
    #     # Foreach particle: register it to a node.
    #     particle_id = 0
    #     while particle_id < num_particles[None]:
    #         # Root as parent,
    #         # 0.5 (center) as the parent centroid position
    #         # 1.0 (whole) as the parent geo size
    #         alloc_a_node_for_particle(particle_id, 0,
    #                                   particle_pos[0] * 0 + 0.5,
    #                                   1.0)
    #
    #         trash_id = 0
    #         while trash_id < trash_table_len[None]:
    #             alloc_a_node_for_particle(trash_particle_id[trash_id],
    #                                       trash_base_parent[trash_id],
    #                                       trash_base_geo_center[trash_id],
    #                                       trash_base_geo_size[trash_id])
    #             trash_id = trash_id + 1
    #
    #         trash_table_len[None] = 0
    #         particle_id = particle_id + 1
    #
    #
    # @ti.func
    # def get_gravity_at(position):
    #     acc = particle_pos[0] * 0
    #
    #     trash_table_len[None] = 0
    #     trash_id = alloc_trash()
    #     assert trash_id == 0
    #     trash_base_parent[trash_id] = 0
    #     trash_base_geo_size[trash_id] = 1.0
    #
    #     trash_id = 0
    #     while trash_id < trash_table_len[None]:
    #         parent = trash_base_parent[trash_id]
    #         parent_geo_size = trash_base_geo_size[trash_id]
    #
    #         particle_id = node_particle_id[parent]
    #         if particle_id >= 0:
    #             distance = particle_pos[particle_id] - position
    #             acc += particle_mass[particle_id] * gravity_func(distance)
    #
    #         else:  # TREE or LEAF
    #             for which in ti.grouped(ti.ndrange(*([2] * DIM))):
    #                 child = node_children[parent, which]
    #                 if child == LEAF:
    #                     continue
    #                 node_center = node_centroid_pos[child] / node_mass[child]
    #                 distance = node_center - position
    #                 if distance.norm_sqr() > \
    #                         SHAPE_FACTOR ** 2 * parent_geo_size ** 2:
    #                     acc += node_mass[child] * gravity_func(distance)
    #                 else:
    #                     new_trash_id = alloc_trash()
    #                     child_geo_size = parent_geo_size * 0.5
    #                     trash_base_parent[new_trash_id] = child
    #                     trash_base_geo_size[new_trash_id] = child_geo_size
    #
    #         trash_id = trash_id + 1
    #
    #     return acc
    #
    #
    # # Helper functions I lifted from 'taichi_glsl'
    # @ti.func
    # def boundReflect(pos, vel, pmin=0, pmax=1, gamma=1, gamma_perpendicular=1):
    #     """
    #     Reflect particle velocity from a rectangular boundary (if collides).
    #     `boundaryReflect` takes particle position, velocity and other parameters.
    #     Detect if the particle collides with the rect boundary given by ``pmin``
    #     and ``pmax``, if collide, returns the velocity after bounced with boundary,
    #     otherwise return the original velocity without any change.
    #     :parameter pos: (Vector)
    #         The particle position.
    #     :parameter vel: (Vector)
    #         The particle velocity.
    #     :parameter pmin: (scalar or Vector)
    #         The position lower boundary. If vector, it's the bottom-left of rect.
    #     :parameter pmax: (scalar or Vector)
    #         The position upper boundary. If vector, it's the top-right of rect.
    #     """
    #     cond = pos < pmin and vel < 0 or pos > pmax and vel > 0
    #     for j in ti.static(range(pos.n)):
    #         if cond[j]:
    #             vel[j] *= -gamma
    #             for k in ti.static(range(pos.n)):
    #                 if k != j:
    #                     vel[k] *= gamma_perpendicular
    #     return vel
    #
    #
    # # The O(NlogN) kernel using quadtree
    # @ti.kernel
    # def substep():
    #     particle_id = 0
    #     while particle_id < num_particles[None]:
    #         acceleration = get_gravity_at(particle_pos[particle_id])
    #         particle_vel[particle_id] += acceleration * DT
    #         # well... seems our tree inserter will break if particle out-of-bound:
    #         particle_vel[particle_id] = boundReflect(
    #             particle_pos[particle_id],
    #             particle_vel[particle_id],
    #             0, 1)
    #         particle_id = particle_id + 1
    #
    #     for i in range(num_particles[None]):
    #         particle_pos[i] += particle_vel[i] * DT
    #
    # %s
    #
    # ''' % (init_func, update_func)

    # --------------------------------------

    def write_to_file(s):
        f = open("created.py", 'w')
        f.write(s)
        f.close()

    # Main of Kernel
    if method == Method.Native:
        write_to_file(raw_kernel_str)
    # elif method == Method.QuadTree:
    #     write_to_file(quadtree_str)

    import created

    created.initialize(2 ** 10)

    return lambda _: created.substep()


@ti.func
def gravity_func(distance):
    l2 = distance.norm_sqr() + 1e-3
    return distance * (l2 ** ((-3) / 2))


if __name__ == '__main__':
    init_circle_str = '''
@ti.kernel
def initialize(num_p: ti.i32):
    for _ in range(num_p):
        particle_id = alloc_particle()

        particle_mass[particle_id] = ti.random() * 1.4 + 0.1

        a = ti.random() * math.tau
        r = ti.sqrt(ti.random()) * 0.3
        particle_pos[particle_id] = 0.5 + ti.Vector([ti.cos(a), ti.sin(a)]) * r
'''

    init_uniform_str = '''
@ti.kernel
def initialize(num_p: ti.i32):
    for _ in range(num_p):
        particle_id = alloc_particle()
        particle_mass[particle_id] = ti.random() * 1.4 + 0.1
        particle_pos[particle_id] = ti.Vector([ti.random(), ti.random()])
'''

    # Pick your ingredient for kernel here
    init = init_circle_str
    update = gravity_func

    kernel = n_body(init, update)

    # Renderer related
    RES = (640, 480)
    gui = ti.GUI('N-body Star', res=RES)

    for _ in range(10000):
        kernel(1)

    # while gui.running:
    #     gui.circles(particle_pos.to_numpy(), radius=2, color=0xfbfcbf)
    #     gui.show()
