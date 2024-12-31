from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygame


@dataclass
class Settings:
    screen_dim: np.ndarray = np.array([1_000, 800], dtype=np.int32)
    max_frame_rate: int = 60

    particle_radius: int = 5
    n_particles: int = 100

    cell_dim: np.ndarray = np.array([100, 100], dtype=np.int32)
    cell_grid: np.ndarray = screen_dim // cell_dim

    velocity_scale: float = 2.0


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
            velocities = settings.velocity_scale * np.random.normal(
                size=(settings.n_particles, 2)
            ).astype(np.float32)
        self.velocities = velocities

        pygame.init()
        self.screen = pygame.display.set_mode(settings.screen_dim)
        pygame.display.set_caption("Particle Simulation")
        self.clock = pygame.time.Clock()

    def simulate_step(self) -> None:
        # Handle particle collisions.
        cells = (self.positions.clip(0, self.settings.screen_dim) // self.settings.cell_dim).astype(
            np.int32
        )

        # TODO: Parallelize cells. GPU?
        for cell_i in range(int(self.settings.cell_grid[0])):
            for cell_j in range(int(self.settings.cell_grid[1])):
                cell = np.array([cell_i, cell_j], dtype=np.int32)
                cell_mask = (cells == cell).all(axis=1)

                # TODO: Check collisions in neighboring cells.
                positions, velocities = self.positions[cell_mask], self.velocities[cell_mask]

                # TODO: 2D elastic collision math.
                for i, pos in enumerate(positions):
                    dists = ((positions[i:] - pos) ** 2.0).sum(axis=-1) ** 0.5
                    too_close = dists < (2 * self.settings.particle_radius)
                    velocities[i:][too_close] = np.roll(velocities[i:][too_close], shift=-1, axis=0)

                self.velocities[cell_mask] = velocities

        # Handle wall collisions.
        # TODO: Only check outer cells.
        too_low = (self.positions - self.settings.particle_radius) < 0
        too_high = (self.positions + self.settings.particle_radius) > self.settings.screen_dim
        reverse_mask = (too_low & (self.velocities < 0)) | (too_high & (self.velocities > 0))
        self.velocities[reverse_mask] *= -1.0

        # Move particles.
        self.positions += self.velocities

    def draw_particles(self) -> None:
        for position in self.positions:
            pygame.draw.circle(
                self.screen, Colors.BLACK, position.astype(np.int32), self.settings.particle_radius
            )

    def simulate(self) -> None:
        cont = True
        while cont:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    cont = False

            self.screen.fill(Colors.WHITE)

            self.simulate_step()
            self.draw_particles()

            pygame.display.flip()
            self.clock.tick(self.settings.max_frame_rate)

    def close(self) -> None:
        pygame.quit()


if __name__ == "__main__":
    settings = Settings()
    particle_container = ParticleContainer(settings)
    particle_container.simulate()
    particle_container.close()
