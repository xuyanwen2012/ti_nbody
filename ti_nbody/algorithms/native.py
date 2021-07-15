@ti.func
def get_raw_gravity_at(pos):
    acc = particle_pos[0] * 0
    for i in range(particle_table_len[None]):
        acc += particle_mass[i] * __GRAVITY_FUNC_NAME__(particle_pos[i] - pos)
    return acc


@ti.kernel
def substep():
    for i in range(particle_table_len[None]):
        acceleration = get_raw_gravity_at(particle_pos[i])
        particle_vel[i] += acceleration * dt
    for i in range(particle_table_len[None]):
        particle_pos[i] += particle_vel[i] * dt