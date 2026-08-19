"""
Visual rendering and video export for GridWorld episodes.

Decoupled from GridWorld itself so training loops never pay rendering
cost; only construct a GridWorldRenderer for eval/demo episodes.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio.v2 as imageio

from GridWorld import Cell, Action

_SPRITE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sprites")
_FOOD_SPRITE_PATH = os.path.join(_SPRITE_DIR, "food.png")
_BACKGROUND_SPRITE_PATH = os.path.join(_SPRITE_DIR, "background.png")

# Per-action fish sprite files. EAT keeps the agent's last facing sprite.
_FISH_SPRITE_PATHS = {
    Action.UP: os.path.join(_SPRITE_DIR, "fish_up.png"),
    Action.DOWN: os.path.join(_SPRITE_DIR, "fish_down.png"),
    Action.LEFT: os.path.join(_SPRITE_DIR, "fish_left.png"),
    Action.RIGHT: os.path.join(_SPRITE_DIR, "fish_right.png"),
}


class GridWorldRenderer:
    """
    Draws GridWorld state as a matplotlib figure (grid + sprites) and can
    record a full episode to an MP4/GIF file.
    """

    def __init__(self, env, cell_px=64, dpi=100):
        self.env = env
        self.cell_px = cell_px
        self.dpi = dpi
        self.figsize = (env.grid_size * cell_px / dpi, env.grid_size * cell_px / dpi)
        self._fish_imgs = {
            action: mpimg.imread(path) for action, path in _FISH_SPRITE_PATHS.items()
        }
        self._facing_action = Action.DOWN  # last movement direction, for EAT/idle frames
        self._food_img = mpimg.imread(_FOOD_SPRITE_PATH)
        self._background_img = mpimg.imread(_BACKGROUND_SPRITE_PATH)

    def _draw(self, ax):
        env = self.env
        n = env.grid_size

        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks(range(n + 1))
        ax.set_yticks(range(n + 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for r in range(n):
            for c in range(n):
                ax.imshow(self._background_img, extent=(c, c + 1, r + 1, r), zorder=0)
        ax.grid(True, color="#ffffff", linewidth=1, alpha=0.4, zorder=1)

        # Food sprites, centered in their cell.
        food_positions = [
            (r, c) for r in range(n) for c in range(n)
            if env.grid[r, c] == Cell.FOOD
        ]
        food_size = 0.5
        for r, c in food_positions:
            ax.imshow(
                self._food_img,
                extent=(c + 0.5 - food_size / 2, c + 0.5 + food_size / 2,
                        r + 0.5 + food_size / 2, r + 0.5 - food_size / 2),
                zorder=2,
            )

        # Agent sprite: fish facing its last movement direction.
        action = getattr(env, "action", None)
        if action in self._fish_imgs:
            self._facing_action = action
        r, c = env.agent_pos
        fish_size = 0.8
        img = self._fish_imgs[self._facing_action]
        ax.imshow(
            img,
            extent=(c + 0.5 - fish_size / 2, c + 0.5 + fish_size / 2,
                    r + 0.5 + fish_size / 2, r + 0.5 - fish_size / 2),
            zorder=3,
        )

        glucose = getattr(env, "current_glucose", env.glucose_start)
        action_name = action.name if action is not None else "-"
        reward = getattr(env, "reward", 0.0)
        ax.set_title(
            f"step {env.step_count}   glucose {glucose}   "
            f"action {action_name}   reward {reward:.2f}",
            fontsize=10,
        )

    def render_frame(self):
        """Render the current env state and return it as an (H, W, 3) uint8 RGB array."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        self._draw(ax)
        fig.tight_layout()
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return frame

    def record_episode(self, policy_fn=None, out_path="episode.mp4", fps=1, max_steps=None):
        """
        Run a full episode from the env's current state, capturing a frame
        per step, and write it to out_path (.mp4 or .gif).

        policy_fn: callable(obs) -> action, or None to use the env's default
        random action sampling (matches GridWorld.step(action=None)).
        """
        frames = [self.render_frame()]
        obs = self.env._get_obs()
        done = False
        steps = 0
        limit = max_steps or self.env.max_steps
        while not done and steps < limit:
            action = policy_fn(obs) if policy_fn is not None else None
            obs, reward, done, info = self.env.step(action)
            frames.append(self.render_frame())
            steps += 1

        imageio.mimsave(out_path, frames, fps=fps)
        return out_path


if __name__ == "__main__":
    env = None
    from GridWorld import GridWorld
    env = GridWorld(seed=6)
    renderer = GridWorldRenderer(env)
    path = renderer.record_episode(out_path="episode.mp4", fps=1)
    print(f"Saved episode video to {path}")
