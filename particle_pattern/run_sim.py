from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import pygame
from PIL import Image
from scipy.signal import convolve2d
from simulation import SimSettings, reverse_velocities, simulate_step


class Colors:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)


def run_game_sim(
    settings: SimSettings, positions: np.ndarray, velocities: np.ndarray, max_frame_rate: int = 120
) -> None:
    """Run the particle simulation interactively as a game."""
    pygame.init()
    screen = pygame.display.set_mode(settings.wall_bounds)
    pygame.display.set_caption("Particle Simulation")
    clock = pygame.time.Clock()

    print("Game sim commands: (q) to quit, (space) to pause, (r) to reverse).")

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
                    velocities = reverse_velocities(velocities)

        screen.fill(Colors.WHITE)
        draw_particles(positions, screen, settings.particle_radius)
        if not pause:
            positions, velocities = simulate_step(
                positions, velocities, settings, delta_time_sec=clock.get_time() / 1000.0
            )
        pygame.display.flip()

        clock.tick(max_frame_rate)
        print(f"FPS={clock.get_fps():.4f}", end="\r")

    pygame.quit()


def draw_particles(positions: np.ndarray, screen: pygame.Surface, particle_radius: int) -> None:
    for position in positions:
        pygame.draw.circle(screen, Colors.BLACK, position.astype(np.int32), particle_radius)


def draw_init_particles(settings: SimSettings, max_frame_rate: int = 120) -> np.ndarray:
    """Initialize particle positions by drawing on screen."""
    positions = np.array([], dtype=np.float32)

    pygame.init()
    screen = pygame.display.set_mode(settings.wall_bounds)
    pygame.display.set_caption("Particle Drawing")
    clock = pygame.time.Clock()

    print("Draw particles commands: (enter) when done.")

    cont = True
    drawing = False

    while cont:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cont = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    cont = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False

        if drawing:
            mouse_pos = np.array(pygame.mouse.get_pos(), dtype=np.float32)
            if len(positions) == 0:
                positions = mouse_pos[None, :]
            else:
                dists = LA.norm(mouse_pos - positions, axis=-1)
                if (dists > (settings.particle_radius * 3.0)).all():
                    positions = np.concat((positions, mouse_pos[None, :]), axis=0)

        screen.fill(Colors.WHITE)
        draw_particles(positions, screen, settings.particle_radius)
        pygame.display.flip()

        clock.tick(max_frame_rate)
        print(f"FPS={clock.get_fps():.4f}", end="\r")

    pygame.quit()
    return positions


def image_init_particles(filename: str, settings: SimSettings, n_particles: int) -> np.ndarray:
    """Initialize particle positions from an image."""
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


def random_init_particles(settings: SimSettings, n_particles: int) -> np.ndarray:
    """Initialize particle positions randomly."""
    positions = np.random.randint(
        low=settings.particle_radius,
        high=settings.wall_bounds - settings.particle_radius,
        size=(n_particles, 2),
    )
    return positions.astype(np.float32)


def get_random_velocities(velocity_scale: float, positions_shape: np.ndarray) -> np.ndarray:
    """Initialize particle velocities randomly."""
    return velocity_scale * np.random.normal(size=positions_shape).astype(np.float32)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--init_type",
        required=False,
        default="draw",
        choices=["draw", "image", "random"],
        help="How the user wants to initalize particles.",
    )
    parser.add_argument(
        "--image_path",
        required=False,
        help="Path of the image to init particles from. Only used if init type is image.",
    )
    parser.add_argument(
        "--n_particles",
        required=False,
        type=int,
        default=500,
        help="Number of particles to init with. Only used for image and random init.",
    )
    parser.add_argument(
        "--velocity_scale",
        required=False,
        type=float,
        default=50.0,
        help="Velocity multiplier (velocity_scale * normal_dist) to init random velocities.",
    )
    parser.add_argument(
        "--output_path",
        required=False,
        default="outputs/particle_sim.mp4",
        help="Output path of the particle simulation animation.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive game mode instead of creating a fixed animation.",
    )
    return parser.parse_args()


def create_animation(
    positions: np.ndarray,
    velocities: np.ndarray,
    settings: SimSettings,
    output_path: Path,
    backward_time: float = 8.0,
    slow_time: float = 3.0,
    slowdown_factor: float = 0.5,
    stop_time: float = 1.0,
    forward_time: float = 7.0,
    frame_rate: int = 30,
) -> None:
    """
    Create an animation for particles making a pattern from seeming randomness.

    BUG: Since the simulation is not perfectly reversible, this animation is fake.
    If we initialized from where the reverse process ended, we would get
    positions that don't equal the original values.
    """
    backward_frames = int(backward_time * frame_rate)
    forward_frames = int(forward_time * frame_rate)
    slow_backward_frames = int(slow_time * frame_rate) // 2
    stop_frames = int(stop_time * frame_rate)
    total_frames = backward_frames + stop_frames + forward_frames

    og_positions, og_velocities = positions.copy(), velocities.copy()
    frames = np.empty((total_frames,) + og_positions.shape, dtype=np.float32)

    normal_delta_time = 1.0 / frame_rate
    slow_delta_time = normal_delta_time * slowdown_factor

    velocities = reverse_velocities(velocities)
    for i in range(backward_frames):
        dt = slow_delta_time if (i < slow_backward_frames) else normal_delta_time
        positions, velocities = simulate_step(positions, velocities, settings, dt)
        frames[backward_frames - 1 - i] = positions.copy()

    positions, velocities = og_positions, og_velocities
    for i in range(stop_frames):
        frames[backward_frames + i] = positions.copy()

    for i in range(forward_frames):
        dt = slow_delta_time if (i < slow_backward_frames) else normal_delta_time
        positions, velocities = simulate_step(positions, velocities, settings, dt)
        frames[backward_frames + stop_frames + i] = positions.copy()

    # Plot animation.
    fig, ax = plt.subplots(figsize=(settings.wall_bounds // 100))
    ax.invert_yaxis()
    ax.set_xlim(0, settings.wall_bounds[0])
    ax.set_ylim(settings.wall_bounds[1], 0)

    img = ax.scatter(
        og_positions[:, 0], og_positions[:, 1], s=settings.particle_radius * 2, c="blue"
    )

    def update_anim(frame: int):
        img.set_offsets(frames[frame])

    animation = anim.FuncAnimation(fig, func=update_anim, frames=total_frames)
    animation.save(output_path, fps=frame_rate)


if __name__ == "__main__":
    args = parse_args()

    settings = SimSettings()

    match args.init_type:
        case "draw":
            positions = draw_init_particles(settings)
        case "image":
            positions = image_init_particles(args.image_path, settings, args.n_particles)
        case "random":
            positions = random_init_particles(settings, args.n_particles)

    velocities = get_random_velocities(args.velocity_scale, positions.shape)

    if args.interactive:
        run_game_sim(settings, positions, velocities)
    else:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        create_animation(positions, velocities, settings, output_path)
