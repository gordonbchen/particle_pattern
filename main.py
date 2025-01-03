from __future__ import annotations

from argparse import ArgumentParser

import numpy as np
import numpy.linalg as LA
import pygame
from PIL import Image
from scipy.signal import convolve2d


class Settings:
    screen_dim: np.ndarray = np.array([1_000, 800], dtype=np.int32)
    max_frame_rate: int = 120

    particle_radius: int = 5
    n_particles: int = 500

    cell_dim: np.ndarray = np.array([100, 100], dtype=np.int32)
    cell_grid: np.ndarray = screen_dim // cell_dim

    velocity_scale: float = 64.0  # pix / sec.


class Colors:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)


class ParticleContainer:
    """A walled container filled with equal mass, equal sized particles that collide elastically."""

    def __init__(
        self,
        settings: Settings,
        positions: np.ndarray | None = None,
        velocities: np.ndarray | None = None,
    ) -> None:
        self.settings = settings

        if positions is None:
            positions = np.random.randint(
                low=settings.particle_radius,
                high=settings.screen_dim - settings.particle_radius,
                size=(settings.n_particles, 2),
            ).astype(np.float32)
        self.positions = positions

        if velocities is None:
            velocities = settings.velocity_scale * np.random.normal(size=positions.shape).astype(
                np.float32
            )
        self.velocities = velocities

        pygame.init()
        self.screen = pygame.display.set_mode(settings.screen_dim)
        pygame.display.set_caption("Particle Simulation")
        self.clock = pygame.time.Clock()

    # TODO: make simulation perfectly reversible.
    def simulate_step(self) -> None:
        # Handle particle collisions.
        cells = (
            (self.positions // self.settings.cell_dim)
            .astype(np.int32)
            .clip(0, self.settings.cell_grid - 1)
        )

        # TODO: Parallelize cells. GPU?
        for cell_i in range(int(self.settings.cell_grid[0])):
            for cell_j in range(int(self.settings.cell_grid[1])):
                cell = np.array([cell_i, cell_j], dtype=np.int32)
                cell_mask = (cells == cell).all(axis=1)

                # TODO: Check collisions in neighboring cells.
                positions, velocities = self.positions[cell_mask], self.velocities[cell_mask]
                self.velocities[cell_mask] = self.collide(positions, velocities)

        # Handle wall collisions.
        outer_mask = ((cells == 0) | (cells == (settings.cell_grid - 1))).any(axis=-1)
        outer_positions, outer_velocities = self.positions[outer_mask], self.velocities[outer_mask]
        too_low = (outer_positions - self.settings.particle_radius) < 0
        too_high = (outer_positions + self.settings.particle_radius) > self.settings.screen_dim

        reverse_mask = (too_low & (outer_velocities < 0)) | (too_high & (outer_velocities > 0))

        outer_velocities[reverse_mask] *= -1.0
        self.velocities[outer_mask] = outer_velocities

        # Move particles.
        self.positions += self.velocities * (self.clock.get_time() / 1_000.0)

    def collide(self, positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        """2d elastic collision physics."""
        dists = LA.norm(positions[:, None] - positions, axis=-1)

        for i, j in zip(*np.where(dists < (2 * self.settings.particle_radius))):
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

    def reverse_velocities(self) -> None:
        self.velocities *= -1

    def draw_particles(self) -> None:
        for position in self.positions:
            pygame.draw.circle(
                self.screen, Colors.BLACK, position.astype(np.int32), self.settings.particle_radius
            )

    def simulate(self) -> None:
        cont = True
        pause = True

        while cont:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    cont = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        pause = not pause
                    if event.key == pygame.K_r:
                        self.reverse_velocities()

            self.screen.fill(Colors.WHITE)
            self.draw_particles()
            if not pause:
                self.simulate_step()
            pygame.display.flip()

            self.clock.tick(self.settings.max_frame_rate)
            print(f"FPS={self.clock.get_fps():.4f}", end="\r")

    def close(self) -> None:
        pygame.quit()

    def calc_kinetic_energy(self) -> float:
        """Since there is no mass, just return sum(|v|^2)."""
        return (LA.norm(self.velocities, axis=-1) ** 2.0).sum()


def image_to_points(filename: str, settings: Settings) -> np.ndarray:
    image = np.array(Image.open(filename).convert("L"), dtype=np.float32) / 255.0

    gaussian_kernel = (
        np.array(
            [
                [2, 4, 5, 4, 2],
                [4, 9, 12, 9, 4],
                [5, 12, 15, 12, 5],
                [4, 9, 12, 9, 4],
                [2, 4, 5, 4, 2],
            ],
            dtype=np.float32,
        )
        / 159
    )
    smoothed = convolve2d(image, gaussian_kernel, mode="valid")

    sobel_v_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    sobel_v_convolved = convolve2d(smoothed, sobel_v_kernel, mode="valid")
    sobel_h_convolved = convolve2d(smoothed, sobel_v_kernel.T, mode="valid")

    convolved = ((sobel_v_convolved**2.0) + (sobel_h_convolved**2.0)) ** 0.5
    image_points = np.stack(
        tuple(reversed(np.where(convolved > np.percentile(convolved, 90)))), axis=-1
    ).astype(np.float32)

    points = image_points[
        np.random.choice(np.arange(len(image_points)), size=settings.n_particles, replace=False)
    ]
    points = (
        (points - points.min(axis=0))
        / (points.max(axis=0) - points.min(axis=0))
        * (settings.screen_dim - (2 * settings.particle_radius))
    ) + settings.particle_radius
    return points


if __name__ == "__main__":
    settings = Settings()

    parser = ArgumentParser()
    parser.add_argument("--image", required=False, help="Path of the image to use.")
    image_file = parser.parse_args().image

    points = image_to_points(image_file, settings) if image_file else None
    particle_container = ParticleContainer(settings, positions=points)
    particle_container.simulate()
    particle_container.close()
