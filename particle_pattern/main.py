from __future__ import annotations

from argparse import ArgumentParser

import numpy as np
import pygame
from PIL import Image
from scipy.signal import convolve2d
from simulation import SimulationSettings, reverse_velocities, simulate_step


class Colors:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)


class ParticleContainer:
    """A walled container filled with equal mass, equal sized particles that collide elastically."""

    def __init__(
        self,
        settings: SimulationSettings,
        positions: np.ndarray,
        velocities: np.ndarray | None = None,
        velocity_scale: float = 32.0,
        max_frame_rate: int = 120,
    ) -> None:
        self.settings = settings
        self.max_frame_rate = max_frame_rate

        self.positions = positions

        if velocities is None:
            velocities = velocity_scale * np.random.normal(size=positions.shape).astype(np.float32)
        self.velocities = velocities

        pygame.init()
        self.screen = pygame.display.set_mode(self.settings.wall_bounds)
        pygame.display.set_caption("Particle Simulation")
        self.clock = pygame.time.Clock()

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
                    elif event.key == pygame.K_q:
                        cont = False
                    elif event.key == pygame.K_r:
                        self.velocities = reverse_velocities(self.velocities)

            self.screen.fill(Colors.WHITE)
            self.draw_particles()
            if not pause:
                self.positions, self.velocities = simulate_step(
                    self.positions,
                    self.velocities,
                    self.settings,
                    delta_time_sec=self.clock.get_time() / 1000.0,
                )
            pygame.display.flip()

            self.clock.tick(self.max_frame_rate)
            print(f"FPS={self.clock.get_fps():.4f}", end="\r")

    def close(self) -> None:
        pygame.quit()


def image_to_points(filename: str, settings: SimulationSettings, n_particles: int) -> np.ndarray:
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

    # TODO: remove overlapping points.
    points = image_points[
        np.random.choice(np.arange(len(image_points)), size=n_particles, replace=False)
    ]
    points = (
        (points - points.min(axis=0))
        / (points.max(axis=0) - points.min(axis=0))
        * (settings.wall_bounds - (2 * settings.particle_radius))
    ) + settings.particle_radius
    return points


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image", required=False, help="Path of the image to use.")
    image_file = parser.parse_args().image

    settings = SimulationSettings()

    n_particles = 500
    if image_file is not None:
        points = image_to_points(image_file, settings, n_particles)
    else:
        points = np.random.randint(
            low=settings.particle_radius,
            high=settings.wall_bounds - settings.particle_radius,
            size=(n_particles, 2),
        ).astype(np.float32)

    particle_container = ParticleContainer(settings, positions=points)
    particle_container.simulate()
    particle_container.close()
