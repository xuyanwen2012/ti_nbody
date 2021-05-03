import taichi as ti


def n_body(init_func, update_func):
    # This is the kernel for raw method
    raw_kernel_str = '''
import taichi as ti
import math

ti.init()

%s 

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
        acc += particle_mass[i] * gravity_func(particle_pos[i] - pos)
    return acc


@ti.kernel
def substep_raw():
    for i in range(num_particles[None]):
        acceleration = get_raw_gravity_at(particle_pos[i])
        particle_vel[i] += acceleration * DT

    for i in range(num_particles[None]):
        particle_pos[i] += particle_vel[i] * DT


%s
    ''' % (init_func, update_func)

    def write_to_file(s):
        f = open("created.py", 'w')
        f.write(s)
        f.close()

    write_to_file(raw_kernel_str)
    import created

    created.initialize(2 ** 10)

    return lambda _: created.substep_raw()


if __name__ == '__main__':
    update_str = '''
@ti.func
def gravity_func(distance):
    """
    Define which ever the equation used to compute gravity here
    :param distance: the distance between things.
    :return:
    """
    # --- The equation defined in the new n-body example
    # (self**2).sum()
    l2 = distance.norm_sqr() + 1e-3
    return distance * (l2 ** ((-3) / 2))
'''

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

    # Maybe define list of particles here ???

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

    # Pick your ingredient for kernel here
    init = declare_tables_str + init_circle_str
    update = update_str
    kernel = n_body(init, update)

    # Renderer related
    RES = (640, 480)
    gui = ti.GUI('N-body Star', res=RES)

    for _ in range(10000):
        kernel(1)

    # while gui.running:
    #     gui.circles(particle_pos.to_numpy(), radius=2, color=0xfbfcbf)
    #     gui.show()
