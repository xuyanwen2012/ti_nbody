import math

import numpy as np
import taichi as ti
from ti_nbody import vortex_rings

eps = 0.01
dt = 0.1


@ti.kernel
def init_tracers(n_tracer: ti.i32):
    for i in range(n_tracer):
        tracer[i] = [ti.random() - 0.5, ti.random() * 3 - 1.5]


@ti.func
def compute_u_single(p, pos_i, vort_i):
    r2 = (p - pos_i).norm() ** 2
    uv = ti.Vector([pos_i.y - p.y, p.x - pos_i.x])
    return vort_i * uv / (r2 * math.pi) * 0.5 * (1.0 - ti.exp(-r2 / eps ** 2))


if __name__ == '__main__':
    # Pick your ingredient for vortex ring here
    init = init_tracers
    update = compute_u_single
    (kernel, tracer) = vortex_rings(init, update)

    # Renderer related
    gui = ti.GUI("Vortex Rings", (1024, 512), background_color=0xFFFFFF)

    while gui.running:
        kernel()

        gui.circles(tracer.to_numpy() * np.array([[0.05, 0.1]]) +
                    np.array([[0.0, 0.5]]),
                    radius=0.5,
                    color=0x0)

        gui.show()
