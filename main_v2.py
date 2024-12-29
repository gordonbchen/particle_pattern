from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygame


@dataclass
class Settings:
    screen_dim: np.ndarray = np.array([1_000, 800])
    max_frame_rate: int = 60

    particle_radius: int = 5
    n_particles: int = 100


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
                low=self.settings.particle_radius,
                high=self.settings.screen_dim - self.settings.particle_radius,
                size=(self.settings.n_particles, 2),
            ).astype(np.float32)
        self.positions = positions

        if velocities is None:
            velocities = np.random.normal(size=(self.settings.n_particles, 2))
        self.velocities = velocities

        pygame.init()
        self.screen = pygame.display.set_mode(settings.screen_dim)
        pygame.display.set_caption("Particle Simulation")
        self.clock = pygame.time.Clock()

    def simulate_step(self) -> None:
        self.positions += self.velocities

        # TODO: 2D elastic collision math.
        # OPTIM: Only check collisions in neighboring cells.
        # OPTIM: Parallelize cells.
        for i, pos in enumerate(self.positions):
            dists = ((self.positions[i + 1 :] - pos) ** 2.0).sum(axis=-1) ** 0.5
            too_close = dists < (2 * self.settings.particle_radius)

        # OPTIM: Only check outer cells.
        for i, bound in enumerate(self.settings.screen_dim):
            too_low = (self.positions[:, i] - self.settings.particle_radius) < 0
            too_high = (self.positions[:, i] + self.settings.particle_radius) > bound
            self.velocities[too_low | too_high] *= -1.0

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

        pygame.quit()


if __name__ == "__main__":
    settings = Settings()
    particle_container = ParticleContainer(settings)
    particle_container.simulate()
