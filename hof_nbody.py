import taichi as ti
import taichi.lang
import math
import sys

ti.init(arch=ti.cpu)
if not hasattr(ti, 'jkl'):
    ti.jkl = ti.indices(1, 2, 3)

# Program related
RES = (640, 480)

# N-body related
DT = 1e-5
DIM = 2
NUM_MAX_PARTICLE = 32768  # 2^15
SHAPE_FACTOR = 1


# ---------------  Utils ---------------------------------------
def declare_particle_tables():
    num = ti.field(dtype=ti.i32, shape=())
    pos = ti.Vector.field(n=DIM, dtype=ti.f32)
    vel = ti.Vector.field(n=DIM, dtype=ti.f32)
    mass = ti.field(dtype=ti.f32)
    table = ti.root.dense(indices=ti.i, dimensions=NUM_MAX_PARTICLE)
    table.place(pos).place(vel).place(mass)

    return num, pos, vel, mass, table


def write_to_file(s):
    f = open("created.py", 'w')
    f.write(s)
    f.close()


@ti.func
def gravity_func(distance):
    """
    Define which ever the equation used to compute gravity here
    :param distance: the distance between things.
    :return:
    """
    l2 = distance.norm_sqr() + 1e-3
    return distance * (l2 ** ((-3) / 2))


@ti.func
def alloc_particle():
    ret = ti.atomic_add(num_particles[None], 1)
    assert ret < NUM_MAX_PARTICLE
    particle_mass[ret] = 0
    particle_pos[ret] = particle_pos[0] * 0
    particle_vel[ret] = particle_pos[0] * 0
    return ret


@ti.kernel
def initialize(num_p: ti.i32):
    """
    Randomly set the initial position of the particles to start with. Note
    set a value to 'num_particles[None]' taichi field to indicate.
    :return: None
    """
    for _ in range(num_p):
        particle_id = alloc_particle()

        m = ti.random() * 1.4 + 0.1
        particle_mass[particle_id] = m

        a = ti.random() * math.tau
        r = ti.sqrt(ti.random()) * 0.3
        x, y = ti.cos(a), ti.sin(a)
        pos = ti.Vector([x, y]) * r
        particle_pos[particle_id] = 0.5 + pos


def n_body(particles, update_func):
    (num_particles, particle_pos, particle_vel,
     particle_mass, particle_table) = particles

    raw_str = '''
import taichi as ti


@ti.func
def get_gravity_at(num_particles, particle_pos, particle_mass, pos):
    acc = particle_pos[0] * 0
    for i in range(num_particles[None]):
        acc += particle_mass[i] * %s(particle_pos[i] - pos)
    return acc

# The O(N^2) kernel algorithm
@ti.kernel
def substep(num_particles: ti.template(), particle_pos: ti.template(), particle_vel: ti.template(), particle_mass: ti.template()):
    for i in range(num_particles[None]):
        acceleration = get_gravity_at(num_particles, particle_pos,particle_mass, particle_pos[i])
        particle_vel[i] += acceleration * DT
        particle_vel[i] = boundReflect(
            particle_pos[i],
            particle_vel[i],
            0, 1)

    for i in range(num_particles[None]):
        particle_pos[i] += particle_vel[i] * DT


@ti.func
def boundReflect(pos, vel, pmin=0, pmax=1, gamma=1, gamma_perpendicular=1):
    """
    Reflect particle velocity from a rectangular boundary (if collides).
    `boundaryReflect` takes particle position, velocity and other parameters.
    Detect if the particle collides with the rect boundary given by ``pmin``
    and ``pmax``, if collide, returns the velocity after bounced with boundary,
    otherwise return the original velocity without any change.
    :parameter pos: (Vector)
        The particle position.
    :parameter vel: (Vector)
        The particle velocity.
    :parameter pmin: (scalar or Vector)
        The position lower boundary. If vector, it's the bottom-left of rect.
    :parameter pmax: (scalar or Vector)
        The position upper boundary. If vector, it's the top-right of rect.
    """
    cond = pos < pmin and vel < 0 or pos > pmax and vel > 0
    for j in ti.static(range(pos.n)):
        if cond[j]:
            vel[j] *= -gamma
            for k in ti.static(range(pos.n)):
                if k != j:
                    vel[k] *= gamma_perpendicular
    return vel

''' % update_func.__name__

    write_to_file(raw_str)
    import created

    desired_substep = taichi.lang.kernel(created.substep)

    return lambda particles: desired_substep(num_particles, particle_pos,
                                             particle_vel, particle_mass)

    # ----------------------- Raw ---------------------------------------------

    # ----------------------- Shared ------------------------------------------

    # Helper functions I lifted from 'taichi_glsl'


if __name__ == '__main__':
    gui = ti.GUI('N-body Star', res=RES)

    # get command line as input
    assert len(sys.argv) == 2
    exp = int(sys.argv[1])

    # Main program starts from here

    particles = declare_particle_tables()

    initialize(2 ** 10)

    update_func = gravity_func
    kernel = n_body(particles, update_func)
    #
    kernel(1)
