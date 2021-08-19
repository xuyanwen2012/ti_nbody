import taichi as ti
import argparse

from ti_nbody import n_body, Method
from ti_nbody.init_functions import uniform


@ti.func
def custom_gravity_func(distance):
    l2 = distance.norm_sqr() + 1e-3
    return distance * (l2 ** ((-3) / 2))


if __name__ == '__main__':
    # Pick your ingredient for ti_nbody here, that's all it is

    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--particles", default=1024, type=int,
                    help="number of particles")

    parser.add_argument("-m", "--mode", default="naive", type=str,
                    help="algorithm mode (tree or naive)")

    parser.add_argument("-t", "--theta", default=1.0, type=float,
                    help="theta parameter for tree mode")

    parser.add_argument("-th", "--threads", default=1, type=int,
                        help="how many threads to run with")    
    
    args = parser.parse_args()

    if args.mode == "naive":
        M = Method.Native
        m = "native"
    elif args.mode == "tree":
        M = Method.QuadTree
        m = "tree"
    else:
        assert(False)

    print("running with particles: " + str(args.particles))
    print("running with method: " + m)
    if m == "tree":        
        print("  tree mode using theta of: " + str(args.theta))
    else:
        print("  running naive mode with threads: " + str(args.threads))
    
    init = (args.particles, uniform)
    update = custom_gravity_func
    (kernel, gen_lib) = n_body(init, update, M, args.threads, args.theta)

    # GUI Renderer related
    RES = (1280, 960)
    gui = ti.GUI('N-body Star', res=RES)

    while gui.running:
        gui.circles(gen_lib.particle_pos.to_numpy(), radius=2, color=0xfbfcbf)
        gui.show()
        kernel()
