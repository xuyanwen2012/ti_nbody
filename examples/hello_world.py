from ti_nbody import n_body

if __name__ == '__main__':
    kernel = n_body(1, 1)

#     init_circle_str = '''
# @ti.ti_nbody
# def initialize(num_p: ti.i32):
#     for _ in range(num_p):
#         particle_id = alloc_particle()
#
#         particle_mass[particle_id] = ti.random() * 1.4 + 0.1
#
#         a = ti.random() * math.tau
#         r = ti.sqrt(ti.random()) * 0.3
#         particle_pos[particle_id] = 0.5 + ti.Vector([ti.cos(a), ti.sin(a)]) * r
# '''
#
#     init_uniform_str = '''
# @ti.ti_nbody
# def initialize(num_p: ti.i32):
#     for _ in range(num_p):
#         particle_id = alloc_particle()
#         particle_mass[particle_id] = ti.random() * 1.4 + 0.1
#         particle_pos[particle_id] = ti.Vector([ti.random(), ti.random()])
# '''
#
#     # Pick your ingredient for ti_nbody here
#     init = init_circle_str
#     update = gravity_func
#
#     kernel = n_body(init, update)
#
#     # Renderer related
#     RES = (640, 480)
#     gui = ti.GUI('N-body Star', res=RES)
#
#     for _ in range(10000):
#         kernel(1)
#
#     # while gui.running:
#     #     gui.circles(particle_pos.to_numpy(), radius=2, color=0xfbfcbf)
#     #     gui.show()
