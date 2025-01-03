import numpy as np
import numpy.linalg as LA


class SimSettings:
    wall_bounds: np.ndarray = np.array([1_000, 800], dtype=np.int32)

    particle_radius: int = 5

    cell_dim: np.ndarray = np.array([100, 100], dtype=np.int32)
    cell_grid: np.ndarray = wall_bounds // cell_dim


def simulate_step(
    positions: np.ndarray, velocities: np.ndarray, settings: SimSettings, delta_time_sec: float
) -> tuple[np.ndarray, np.ndarray]:
    """Make a simulation step. Return new positions and velocities."""
    # TODO: make simulation perfectly reversible.
    # Handle particle collisions.
    cells = (positions // settings.cell_dim).astype(np.int32).clip(0, settings.cell_grid - 1)

    # TODO: Parallelize cells. GPU?
    for cell_i in range(int(settings.cell_grid[0])):
        for cell_j in range(int(settings.cell_grid[1])):
            cell = np.array([cell_i, cell_j], dtype=np.int32)
            cell_mask = (cells == cell).all(axis=1)

            # TODO: Check collisions in neighboring cells.
            cell_positions, cell_velocities = positions[cell_mask], velocities[cell_mask]
            velocities[cell_mask] = collide(
                cell_positions, cell_velocities, settings.particle_radius
            )

    # Handle wall collisions.
    outer_mask = ((cells == 0) | (cells == (settings.cell_grid - 1))).any(axis=-1)
    outer_positions, outer_velocities = positions[outer_mask], velocities[outer_mask]
    too_low = (outer_positions - settings.particle_radius) < 0
    too_high = (outer_positions + settings.particle_radius) > settings.wall_bounds

    reverse_mask = (too_low & (outer_velocities < 0)) | (too_high & (outer_velocities > 0))

    outer_velocities[reverse_mask] *= -1.0
    velocities[outer_mask] = outer_velocities

    # Move particles.
    positions += velocities * delta_time_sec
    return positions, velocities


def collide(positions: np.ndarray, velocities: np.ndarray, particle_radius: float) -> np.ndarray:
    """2d elastic collision physics. Returns velocities after handling collisions."""
    dists = LA.norm(positions[:, None] - positions, axis=-1)

    for i, j in zip(*np.where(dists < (2 * particle_radius))):
        if j <= i:
            continue

        unit_vector = (positions[i] - positions[j]) / (dists[i, j] + np.finfo(np.float32).eps)
        rel_velocity = (velocities[i] - velocities[j]).dot(unit_vector)

        # Only handle collision if particles are moving towards each other.
        if rel_velocity < 0:
            new_parallel = rel_velocity * unit_vector
            velocities[i] -= new_parallel
            velocities[j] += new_parallel

    return velocities


def reverse_velocities(velocities: np.ndarray) -> np.ndarray:
    """Reverse particle velocities."""
    return velocities * -1.0


def calc_kinetic_energy(velocities: np.ndarray) -> float:
    """Since there is no mass, just return sum(|v|^2)."""
    return (LA.norm(velocities, axis=-1) ** 2.0).sum()
