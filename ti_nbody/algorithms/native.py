@ti.func
def get_raw_gravity_at(pos):
    acc = particle_pos[0] * 0
    for i in range(particle_table_len[None]):
        acc += particle_mass[i] * __GRAVITY_FUNC_NAME__(particle_pos[i] - pos)
    return acc


@ti.kernel
def substep():
    ti.parallelize(kNumThreads)
    for i in range(particle_table_len[None]):
        acceleration = get_raw_gravity_at(particle_pos[i])
        particle_vel[i] += acceleration * dt
        # We dont have to do it here, but we want to make sure the results is
        # same with the quadtree approach.
        particle_vel[i] = reflect_boundary(particle_pos[i],
                                           particle_vel[i],
                                           0, 1)
    for i in range(particle_table_len[None]):
        particle_pos[i] += particle_vel[i] * dt
