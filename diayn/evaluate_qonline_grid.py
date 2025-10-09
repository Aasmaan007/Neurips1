import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import gym_windy_gridworlds
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap

# --------- Define QNetwork (same as your training) ----------
class QNetwork(nn.Module):
    def __init__(self, env, nskills):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod() + nskills, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)
    
def _ensure_per_skill(coord, nskills):
    if isinstance(coord, (list, tuple)) and len(coord) == 2 and isinstance(coord[0], (int, np.integer)):
        return [tuple(coord)] * nskills
    if isinstance(coord, (list, tuple)) and len(coord) == nskills:
        return [tuple(c) for c in coord]
    raise ValueError("starts/goals must be (row,col) or list of length nskills of (row,col).")

def plot_windy_trajectories_combined_actions(
    trajectories,              # list[nskills] -> list[episodes] -> list[(state, action)]
    grid_size,                 # N for NxN grid
    wind_strengths,            # length == grid_size (per column)
    starts,                    # (row,col) or list per skill
    goals,                     # (row,col) or list per skill
    save_path="skills_all.png",
    grid_dot_color="#9e9e9e",
    arrow_width=0.012,
    head_w=0.30,
    head_l=0.30
):
    nskills = len(trajectories)

    wind_strengths = np.asarray(wind_strengths, dtype=float)
    if wind_strengths.shape[0] != grid_size:
        raise ValueError(f"wind_strengths length {wind_strengths.shape[0]} must equal grid_size {grid_size}.")
    starts = _ensure_per_skill(starts, nskills)
    goals  = _ensure_per_skill(goals,  nskills)

    wind_matrix = np.tile(wind_strengths[None, :], (grid_size, 1))
    vmax = float(max(wind_strengths.max(), 1.0))

    tab10 = get_cmap("tab10")
    colors = [tab10(i % 10) for i in range(nskills)]

    # action -> delta(row, col)
    a2d = {0: (-1, 0), 1: (0, +1), 2: (+1, 0), 3: (0, -1)}

    # helper: clamp arrow end so it never crosses the plot boundary
    def clamp_endpoint(x0, y0, x1, y1, gx):
        x_min, x_max = -0.5, gx - 0.5
        y_min, y_max = -0.5, gx - 0.5
        x1 = min(max(x1, x_min), x_max)
        y1 = min(max(y1, y_min), y_max)
        return x1, y1

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("All Skills Trajectories")

    im = ax.imshow(
        wind_matrix, cmap="Blues", vmin=0, vmax=vmax,
        origin="upper",
        extent=[-0.5, grid_size - 0.5, grid_size - 0.5, -0.5],
        interpolation="nearest",
        zorder=0
    )

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)  # row 0 at top
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.grid(True, which="both", color="#bbbbbb", linewidth=0.6)
    ax.set_aspect("equal")

    for spine in ax.spines.values():
        spine.set_edgecolor("#222222")
        spine.set_linewidth(1.2)

    # dots at ALL states
    all_cols, all_rows = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    ax.scatter(all_cols.flatten(), all_rows.flatten(), s=6, c=grid_dot_color, zorder=2)

    # draw per-skill arrows; mark E at the LAST STATE of each episode
    for skill_id, skill_trajs in enumerate(trajectories):
        col = colors[skill_id]
        for episode in skill_trajs:
            if len(episode) == 0:
                continue
            # E should be placed at the last state's coordinate
            last_state = episode[-1][0]  # (row, col) of the final state
            # arrows
            for (state, action) in episode:
                r0, c0 = state
                if action not in a2d:
                    continue
                dr, dc = a2d[action]
                # desired (possibly out-of-grid) end (cell center step)
                x0, y0 = c0, r0
                x1, y1 = c0 + dc, r0 + dr
                # clamp to plot boundary so the arrow stops at the edge
                x1c, y1c = clamp_endpoint(x0, y0, x1, y1, grid_size)
                dx, dy = (x1c - x0), (y1c - y0)

                ax.arrow(
                    x0, y0, dx, dy,
                    width=arrow_width, head_width=head_w, head_length=head_l,
                    fc=col, ec=col, alpha=0.95,
                    length_includes_head=True, zorder=4,
                    clip_on=True  # ensure nothing draws outside
                )

            # mark E at the last state (where the last action is taken)
            er, ec = last_state
            ax.text(
                ec, er, "E", ha="center", va="center",
                fontsize=11, color="black", fontweight="bold",
                zorder=6, clip_on=True
            )

    # mark S/G (single or per-skill)
    unique_starts = len({s for s in starts}) == 1
    unique_goals  = len({g for g in goals}) == 1

    if unique_starts and unique_goals:
        s_r, s_c = starts[0]
        g_r, g_c = goals[0]
        ax.text(s_c, s_r, "S", ha="center", va="center",
                fontsize=12, color="green", fontweight="bold", zorder=6)
        ax.text(g_c, g_r, "G", ha="center", va="center",
                fontsize=12, color="black", fontweight="bold", zorder=6)
    else:
        for skill_id, (s, g) in enumerate(zip(starts, goals)):
            s_r, s_c = s
            g_r, g_c = g
            col = colors[skill_id]
            ax.text(s_c, s_r, f"S{skill_id}", ha="center", va="center",
                    fontsize=9, color=col, fontweight="bold", zorder=6)
            ax.text(g_c, g_r, f"G{skill_id}", ha="center", va="center",
                    fontsize=9, color=col, fontweight="bold", zorder=6)

    # legend (bottom-right) includes E
    skill_handles = [Line2D([0], [0], color=colors[i], lw=2, label=f"Skill {i}") for i in range(nskills)]
    sg_handles = [
        Line2D([0], [0], marker='$S$', color='none', label='S: Start', markerfacecolor='green', markersize=12),
        Line2D([0], [0], marker='$G$', color='none', label='G: Goal', markerfacecolor='black', markersize=12),
        Line2D([0], [0], marker='$E$', color='none', label='E: End (last state)', markerfacecolor='black', markersize=12),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=grid_dot_color, markersize=5, label='All states'),
    ]
    leg = ax.legend(handles=skill_handles + sg_handles, loc="lower right", frameon=True)
    for legobj in leg.legendHandles:
        try:
            legobj.set_linewidth(2.0)
        except Exception:
            pass

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Wind strength (upwards)", rotation=90)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved combined plot → {save_path}")   

# --------- Load Model Weights ----------
def load_model(pth_file, env, nskills):
    model = QNetwork(env, nskills)
    model.load_state_dict(torch.load(pth_file, map_location="cpu"))
    model.eval()
    return model

def plot_windy_trajectories_separate(
    trajectories,
    grid_size,
    wind_strengths,
    starts,              # (row,col) or list of length nskills
    goals,               # (row,col) or list of length nskills
    save_prefix="skill_traj",
    arrow_color="#8B0000",    # dark red
    grid_line="#222222"       # low-intensity black for border
):
    """
    trajectories: list[nskills] -> list[episodes] -> list[(obs, next_obs)]
                  obs/next_obs are (row, col)
    grid_size: N for an N x N grid
    wind_strengths: list/array length == grid_size (per column upward wind)
    starts/goals: (row,col) or list per-skill
    """
    nskills = len(trajectories)

    # normalize inputs
    wind_strengths = np.asarray(wind_strengths, dtype=float)
    if wind_strengths.shape[0] != grid_size:
        raise ValueError(f"wind_strengths length {wind_strengths.shape[0]} must equal grid_size {grid_size}")

    starts = _ensure_per_skill(starts, nskills)
    goals  = _ensure_per_skill(goals,  nskills)

    # prepare wind background (repeat column strengths across rows)
    wind_matrix = np.tile(wind_strengths[None, :], (grid_size, 1))
    vmax = float(max(wind_strengths.max(), 1.0))  # avoid zero-range

    for skill_id, skill_trajs in enumerate(trajectories):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(f"Skill {skill_id}")

        # wind background: darker blue = stronger upward wind (row 0 at top)
        im = ax.imshow(
            wind_matrix,
            cmap="Blues",
            vmin=0, vmax=vmax,
            origin="upper",
            extent=[-0.5, grid_size - 0.5, grid_size - 0.5, -0.5],
            interpolation="nearest",
            zorder=0
        )

        # grid & axes
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(grid_size - 0.5, -0.5)  # flip y so row 0 is top
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.grid(True, which="both", color="#bbbbbb", linewidth=0.6)
        ax.set_aspect("equal")

        # subtle border color
        for spine in ax.spines.values():
            spine.set_edgecolor(grid_line)
            spine.set_linewidth(1.2)

        # draw trajectory arrows (dark red), interpreting (row,col) -> (x=col, y=row)
        for episode in skill_trajs:
            for (obs, nxt) in episode:
                r0, c0 = obs
                r1, c1 = nxt
                x0, y0 = c0, r0
                dx, dy = (c1 - c0), (r1 - r0)
                ax.arrow(
                    x0, y0, dx, dy,
                    head_width=0.30, head_length=0.30,
                    fc=arrow_color, ec=arrow_color, alpha=0.9,
                    length_includes_head=True, zorder=3
                )

        # place S and G supplied by caller
        s_r, s_c = starts[skill_id]
        g_r, g_c = goals[skill_id]
        ax.text(s_c, s_r, "S", ha="center", va="center",
                fontsize=12, color="green", fontweight="bold", zorder=4)
        ax.text(g_c, g_r, "G", ha="center", va="center",
                fontsize=12, color="black", fontweight="bold", zorder=4)

        # legend (S, G, and arrow meaning)
        legend_handles = [
            Line2D([0], [0], marker='$S$', color='none', label='S: Start',
                   markerfacecolor='green', markersize=12),
            Line2D([0], [0], marker='$G$', color='none', label='G: Goal',
                   markerfacecolor='black', markersize=12),
            Line2D([0], [0], color=arrow_color, lw=2, label='Trajectory (action steps)'),
        ]
        leg = ax.legend(handles=legend_handles, loc="upper right", frameon=True)
        for legobj in leg.legendHandles:
            try:
                legobj.set_linewidth(2.0)
            except Exception:
                pass

        # colorbar to explain wind strengths
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Wind strength (upwards)", rotation=90)

        plt.tight_layout()
        out_path = f"{save_prefix}_{skill_id}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved: {out_path}")


# --------- Rollout Function ----------
def generate_trajectories(env, model, nskills, max_steps=500, n_episodes=1):
    """
    Generate trajectories for each skill (one-hot).
    Returns: list of trajectories, one per skill
    Each trajectory = list of (obs, skill, action, reward, next_obs, done)
    """
    trajectories = []
    obs_dim = np.array(env.observation_space.shape).prod()

    for skill in range(nskills):
        skill_vec = np.zeros(nskills, dtype=np.float32)
        skill_vec[skill] = 1.0

        skill_trajectories = []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            traj = []
            for step in range(max_steps):
                obs_flat = obs.flatten()
                inp = np.concatenate([obs_flat, skill_vec], axis=0)
                inp = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    q_values = model(inp)
                    action = torch.argmax(q_values, dim=1).item()

                next_obs, reward, done, trunc, info = env.step(action)

                obs_listing = obs.astype(np.int32).tolist()
                next_obs_listing = next_obs.astype(np.int32).tolist()
                traj.append((obs_listing, action))

                obs = next_obs
                if done or trunc:
                    break
            skill_trajectories.append(traj)

        trajectories.append(skill_trajectories)
    return trajectories

# --------- Example Usage ----------
if __name__ == "__main__":
    env = gym.make("WindyGridWorld-v0")  # <-- replace with your env
    nskills = 6
    model = "runs/checkpoints/qtargetmaml/WindyGridWorld-v0__q_online__1__2025-07-09_14-22-30__1752051150/latest.pth"
    
    checkpoint = torch.load(model)
    q_network = QNetwork(env, nskills)
    q_network.load_state_dict(checkpoint["q_network_state_dict"])

    trajectories = generate_trajectories(env, q_network, nskills, max_steps=200, n_episodes=1)

    # trajectories is a list of 6 entries, each with a trajectory for that skill
    print(f"Generated {len(trajectories)} skills' trajectories")
    for i, traj in enumerate(trajectories):
        print(f"Skill {i}, Trajectory length: {len(traj[0])}")

    print(trajectories)
    wind_10 = [2, 0, 1, 2, 1, 0, 0, 1, 2, 2]
starts  = (3, 0)  # scalar => same S for all skills (drawn once)
goals   = (3, 7)  # scalar => same G for all skills

plot_windy_trajectories_combined_actions(
    trajectories,                # your list[6] of episodes of (state, action)
    grid_size=len(wind_10),
    wind_strengths=wind_10,
    starts=starts,
    goals=goals,
    save_path="skills_all.png"
)