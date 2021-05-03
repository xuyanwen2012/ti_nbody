import taichi as ti
import taichi.lang

ti.init(arch=ti.cpu)


def write_to_file(s):
    f = open("created.py", 'w')
    f.write(s)
    f.close()


DT = 1e-5
DIM = 2
NUM_MAX_PARTICLE = 1024
num_particles = ti.field(dtype=ti.i32, shape=())
particle_pos = ti.Vector.field(n=DIM, dtype=ti.f32)
table = ti.root.dense(indices=ti.i, dimensions=NUM_MAX_PARTICLE)
table.place(particle_pos)


@ti.func
def alloc_particle():
    ret = ti.atomic_add(num_particles[None], 1)
    assert ret < NUM_MAX_PARTICLE
    particle_pos[ret] = particle_pos[0] * 0
    return ret


@ti.kernel
def initialize(num_p: ti.i32):
    for _ in range(num_p):
        particle_id = alloc_particle()
        particle_pos[particle_id] = ti.Vector([0.5, 0.5])


@ti.func
def gravity_func(distance):
    """
    Define which ever the equation used to compute gravity here
    :param distance: the distance between things.
    :return:
    """
    l2 = distance.norm_sqr() + 1e-3
    return distance * (l2 ** ((-3) / 2))


if __name__ == '__main__':
    k_str = '''
import taichi as ti

@ti.func
def get_gravity_at(n_par, par_pos, pos):
    acc = par_pos[0] * 0
    for i in range(n_par[None]):
        acc += gravity_func(par_pos[i] - pos)
    return acc


# The O(N^2) kernel algorithm
@ti.kernel
def substep(n_par: ti.template(), par_pos: ti.template()):
    for i in range(n_par[None]):
        acceleration = get_gravity_at(n_par, par_pos, par_pos[i])
        particle_pos[i] += 0.001 * acceleration
    '''
    write_to_file(k_str)
    import created

    initialize(2 ** 10)

    test_kernel = taichi.lang.kernel(created.substep)

    test_kernel(num_particles, particle_pos)

    # print(test_kernel())
