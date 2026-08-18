"""
Visual rendering and video export for GridWorld episodes.

Decoupled from GridWorld itself so training loops never pay rendering
cost; only construct a GridWorldRenderer for eval/demo episodes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Circle
import imageio.v2 as imageio

from GridWorld import Cell, Action

# Direction (in degrees) the agent-triangle points for each action.
# EAT keeps the agent's last facing direction.
_FACING_DEG = {
    Action.UP: 180,
    Action.RIGHT: -90,
    Action.DOWN: 0,
    Action.LEFT: 90,
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
        self._facing_deg = 0  # last movement direction, for EAT/idle frames

    def _draw(self, ax):
        env = self.env
        n = env.grid_size

        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks(range(n + 1))
        ax.set_yticks(range(n + 1))
        ax.grid(True, color="#dddddd", linewidth=1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Food sprites: small green circles centered in their cell.
        food_positions = [
            (r, c) for r in range(n) for c in range(n)
            if env.grid[r, c] == Cell.FOOD
        ]
        for r, c in food_positions:
            ax.add_patch(Circle((c + 0.5, r + 0.5), radius=0.18, color="#4caf50", zorder=2))

        # Agent sprite: a triangle pointing in its last movement direction.
        action = getattr(env, "action", None)
        if action in _FACING_DEG:
            self._facing_deg = _FACING_DEG[action]
        r, c = env.agent_pos
        agent = RegularPolygon(
            (c + 0.5, r + 0.5), numVertices=3, radius=0.32,
            orientation=np.radians(self._facing_deg),
            color="#1e88e5", zorder=3,
        )
        ax.add_patch(agent)

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
