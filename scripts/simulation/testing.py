import jax.numpy as jnp
import jax


# Discretizing space, Time, and velocities
N_X = 400
N_Y = 100
DELTA_T = 0.05 #seconds
DELTA_X = 1 #1 pixel delta X
N_DVELOCITIES = 9 #number of discrete velocities
SOUND_SPEED = DELTA_X / (DELTA_T*jnp.sqrt(3))
RIGHT_VELOCITY = 2.5


LATTICE_VELOCITIES = jnp.array([
    [0, 1, 0, -1, 0, 1, -1, -1, 1],
    [0, 0, 1, 0, -1, 1, 1, -1, -1]
])

LATTICE_WEIGHTS = jnp.array([4, 1, 1, 1, 1, 0.25, 0.25, 0.25, 0.25]) / 9



def get_density(discrete_velocities):
    # The sum of all discrete velocities in that square
    return jnp.sum(discrete_velocities, axis=-1)

def get_macroscopic_velocities(density, discrete_velocities):
    return jnp.einsum(
        "NMQ,dQ->NMd", 
        discrete_velocities,
        LATTICE_VELOCITIES
    ) / density[..., jnp.newaxis] #broadcasting to 3D

def get_equilibrium_velocities(macroscopic_velocities, density):
    projected_discrete_vel = jnp.einsum(
        "dQ,NMd->NMQ",
        LATTICE_VELOCITIES, 
        macroscopic_velocities
    )

    macro_velocity_magnitude = jnp.linalg.norm(
        macroscopic_velocities,
        axis=-1, 
        ord=2
    )
    equilibrium_discrete_vel = (
                                density[..., jnp.newaxis] * LATTICE_WEIGHTS[jnp.newaxis, jnp.newaxis, :] 
                                * (1 + 3*projected_discrete_vel + 9/2 * (projected_discrete_vel**2) 
                                   - 3/2 * (macro_velocity_magnitude[..., jnp.newaxis]**2))
    )
    return equilibrium_discrete_vel


if __name__ == "__main__":
    velocity_profile = jnp.zeros((N_X, N_Y, 2))
    velocity_profile = velocity_profile.at[:, :, 0].set(RIGHT_VELOCITY)
    discrete_velocities_prev = get_equilibrium_velocities(velocity_profile, jnp.ones((N_X, N_Y)))
    print(discrete_velocities_prev)
    # print(discrete_velocities_prev.shape)