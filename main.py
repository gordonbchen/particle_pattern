from __future__ import annotations

import time
import turtle
from argparse import ArgumentParser
from random import randint

import numpy as np


class SimArea:
    def __init__(
        self,
        width: int,
        height: int,
        cell_grid: tuple[int, int] = (1, 1),
        title: str = "Sim",
        color: str = "white",
    ) -> None:
        self.width = width
        self.height = height

        self.x_bounds = (-width // 2, width // 2)
        self.y_bounds = (-height // 2, height // 2)

        self.screen = turtle.Screen()
        self.screen.setup(width, height)
        self.screen.tracer(0)
        self.screen.bgcolor(color)
        self.screen.title(title)

        self.cell_grid = cell_grid
        self.cell_width = self.width / cell_grid[0]
        self.cell_height = self.height / cell_grid[1]
        self.cells = {(i, j): set() for j in range(cell_grid[1]) for i in range(cell_grid[0])}

    def update(self) -> None:
        self.screen.update()

    def add_particle(self, pos: np.ndarray | None = None) -> None:
        particle = Particle(self, pos=pos)
        self.update_cell(particle)

    def update_cell(self, particle: Particle) -> None:
        new_cell = self.get_cell(particle.pos[0], particle.pos[1])
        if new_cell != particle.cell:
            if particle.cell is not None:
                self.cells[particle.cell].remove(particle)

            self.cells[new_cell].add(particle)
            particle.set_cell(new_cell)

    def get_cell(self, x: float, y: float) -> tuple[int, int]:
        cell_i = int((x - self.x_bounds[0]) // self.cell_width)
        cell_j = int((y - self.y_bounds[0]) // self.cell_height)

        cell_i = self._clamp(cell_i, 0, self.cell_grid[0] - 1)
        cell_j = self._clamp(cell_j, 0, self.cell_grid[1] - 1)
        return cell_i, cell_j

    def _clamp(self, x: int, lower: int, upper: int) -> int:
        return min(max(lower, x), upper)

    def simulate_step(self) -> None:
        for particles in self.cells.values():
            particles = list(particles)
            for i, particle in enumerate(particles):
                particle.update_pos()

                # TODO: Check collisions in neighboring cells.
                for other_particle in particles[i + 1 :]:
                    particle.collide(other_particle)

        self.update()

    def draw(self) -> None:
        print("Left click to add particle. Right click to QUIT.")

        self.screen.onclick(
            lambda x, y: self.add_particle(np.array([x, y], dtype=np.float32)), btn=1
        )

        self.cont_draw = True

        def quit_draw(*args) -> None:
            self.cont_draw = False

        self.screen.onclick(quit_draw, btn=3)

        while self.cont_draw:
            self.update()
            time.sleep(0.05)

        del self.cont_draw

    def close(self) -> None:
        self.screen.bye()


class Particle:
    def __init__(
        self,
        sim_area: SimArea,
        pos: np.ndarray | None = None,
        vel: np.ndarray | None = None,
        color: str = "black",
        size_factor: float = 0.5,
    ) -> None:
        self.sim_area = sim_area
        self.cell = None

        self.turtle = turtle.Turtle()
        self.turtle.shape("circle")
        self.turtle.color(color)
        self.turtle.penup()

        self.turtle.shapesize(size_factor)
        self.radius = (20 / 2) * size_factor  # Default diameter is 20 pixels.

        self.pos = (
            np.array(
                [
                    randint(sim_area.x_bounds[0] + self.radius, sim_area.x_bounds[1] - self.radius),
                    randint(sim_area.y_bounds[0] + self.radius, sim_area.y_bounds[1] - self.radius),
                ],
                dtype=np.float32,
            )
            if pos is None
            else pos
        )
        self.set_pos(self.pos)
        self.vel = np.random.normal(0, 1.5, size=2) if vel is None else vel

    def set_pos(self, pos: np.ndarray) -> None:
        self.turtle.setpos(pos)

    def update_pos(self) -> None:
        self.pos += self.vel
        self.set_pos(self.pos)
        self.sim_area.update_cell(self)
        self.collide_container()

    def collide_container(self) -> None:
        if not self._coord_in_bounds(self.pos[0], *self.sim_area.x_bounds):
            self.vel[0] *= -1
        if not self._coord_in_bounds(self.pos[1], *self.sim_area.y_bounds):
            self.vel[1] *= -1

    def _coord_in_bounds(self, x: float, lower: int, upper: int) -> bool:
        return (x - self.radius >= lower) and (x + self.radius <= upper)

    def set_cell(self, new_cell: tuple[int, int]) -> None:
        self.cell = new_cell

    def collide(self, other: Particle) -> None:
        """Assumes equal mass particles and head-on elastic collisions."""
        # TODO: True 2D elastic collisions. Fix sticking problem.
        dist = ((self.pos - other.pos) ** 2.0).sum() ** 0.5
        if (self.radius + other.radius) > dist:
            self.vel, other.vel = other.vel, self.vel


def parse_args() -> str:
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        default="random",
        choices=["random", "draw", "image"],
        help="Add random particles.",
    )

    args = parser.parse_args()
    return args.mode


if __name__ == "__main__":
    mode = parse_args()

    sim_area = SimArea(width=1_000, height=800, cell_grid=(10, 8), title="Particle Simulation")

    if mode == "random":
        for i in range(400):
            sim_area.add_particle()
    elif mode == "draw":
        sim_area.draw()
    elif mode == "image":
        # TODO: add image conv mode.
        pass

    t0 = time.time()
    SIM_STEPS = 1_024
    for i in range(SIM_STEPS):
        sim_area.simulate_step()  # TODO: control frame rate.
    sim_area.close()

    elapsed_time = time.time() - t0
    print(f"Elapsed time: {elapsed_time} sec")
    print(f"Frame rate: {SIM_STEPS / elapsed_time} fps")


# TODO: only particle container class.