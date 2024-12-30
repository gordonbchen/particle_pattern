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
        # TODO: 2D elastic collision math.
        # TODO: Only check collisions in neighboring cells.
        # TODO: Parallelize cells.
        for i, pos in enumerate(self.positions):
            for j, other_pos in enumerate(self.positions[i + 1 :]):
                dist = ((pos - other_pos) ** 2.0).sum() ** 0.5
                if dist < (2 * self.settings.particle_radius):
                    self.velocities[[i, i + 1 + j]] = self.velocities[[i + 1 + j, i]]

        # TODO: Only check outer cells.
        too_low = (self.positions - self.settings.particle_radius) < 0
        too_high = (self.positions + self.settings.particle_radius) > self.settings.screen_dim
        reverse_mask = (too_low & (self.velocities < 0)) | (too_high & (self.velocities > 0))
        self.velocities[reverse_mask] *= -1.0

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
