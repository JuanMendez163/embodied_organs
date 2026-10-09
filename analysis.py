"""
Episode logging, training/evaluation, and plotting helpers for Q-learning experiments.

Every function takes what it needs as arguments (env/agent parameter dicts, window sizes,
output paths) and never reads notebook globals, so notebook sections can run independently.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

from GridWorld import GridWorld, Action
from QLearningAgent import QLearningAgent

# ----------------------------------------------------------
# Plot constants shared across figures
# ----------------------------------------------------------

# action categories used in every action plot; each keeps its color across figures
ACTION_CATS = ["move", "eat", "idle"]
CAT_COLORS = {"move": "#2a78d6", "eat": "#eb6834", "idle": "#1baf7a"}
STRIP_CATS = ["eat", "idle"] # move is every other step, so it's left out of the trajectory action strip

LINE_COLOR = "#2a78d6"
LIGHT_LINE_COLOR = "#97bcea"
FOOD_LEFT_COLOR = "#8c8b85"
# light end of "Blues" is too pale to see on white, so start partway in
STEP_CMAP = LinearSegmentedColormap.from_list("steps", plt.get_cmap("Blues")(np.linspace(0.3, 1, 256)))


# ----------------------------------------------------------
# Running episodes
# ----------------------------------------------------------

def classify_step(action):
    """
    Maps one step to an action category (see ACTION_CATS).
    """
    if action == Action.EAT:
        return "eat"
    if action == Action.IDLE:
        return "idle"
    return "move"


def run_episode(env, agent, learn=True):
    """
    Runs one episode. If learn=True, updates the Q-table (same as train()).
    Returns a dict of per-step arrays. glucose, positions, food_left, and states have
    their start-of-episode value at index 0, so they have one more entry than
    reward/actions. Step t (1-based) corresponds to index t-1 of reward/actions.
    """
    obs = env.reset() # reset environment
    state = agent.discretize_state(obs) # convert observation to state for q-learning
    # copy: obs["food_positions"] is env.food_indices itself, which step() edits in place
    food_start = list(obs["food_positions"])
    glucose, rewards, actions = [obs["glucose_level"]], [], []
    positions = [obs["agent_pos"]] # positions[t] is the agent's tile after step t
    food_left = [len(food_start)]
    categories = []
    states = [state] # states[t] is the discretized state after step t
    eaten_at = {} # food tile -> step it was eaten
    done = False
    info = None
    while not done: # run until episode ends
        action = agent.choose_action(state, obs["valid_actions"]) # only actions allowed on this tile
        pos_before = env.agent_pos
        next_obs, reward, done, info = env.step(action) # take a step based on action
        next_state = agent.discretize_state(next_obs)
        if learn: # update Q-table if learning
            starved = next_obs["glucose_level"] == 0 # terminal; ending at max_steps is not
            agent.update(state, action, reward, next_state, next_obs["valid_actions"], terminal=starved)
        state = next_state
        obs = next_obs
        states.append(state)

        glucose.append(next_obs["glucose_level"])
        rewards.append(reward)
        actions.append(int(action)) # we store all actions taken too
        positions.append(next_obs["agent_pos"])
        food_left.append(len(next_obs["food_positions"]))
        categories.append(classify_step(action))
        if action == Action.EAT:
            eaten_at[pos_before] = len(actions) # EAT doesn't move the agent, so this is the eaten tile
    return {
        "glucose": np.array(glucose),
        "reward": np.array(rewards),
        "cum_reward": np.cumsum(rewards), # adds all the rewards up to this step
        "actions": np.array(actions),
        "categories": np.array(categories),
        "positions": np.array(positions), # (row, col)
        "food_start": food_start,
        "food_left": np.array(food_left),
        "eaten_at": eaten_at,
        "states": states,
        "epsilon": agent.epsilon,
        "info": info,
    }


def action_counts(traj):
    """
    Number of steps in each action category for one episode.
    """
    return {cat: int(np.sum(traj["categories"] == cat)) for cat in ACTION_CATS}


def describe_episode(label, traj):
    """
    One-line text summary of an episode, for printing.
    """
    counts = action_counts(traj)
    return (f"{label}: {len(traj['reward'])} steps, total reward {traj['reward'].sum():.1f}, "
            f"eps={traj['epsilon']:.3f}, ate {counts['eat']}/{len(traj['food_start'])}, "
            f"idle {counts['idle']}, move {counts['move']}  ({traj['info']})")


# ----------------------------------------------------------
# Training and evaluation
# ----------------------------------------------------------

def train_agent(env_params, agent_params, n_episodes, seed=None, record_episodes=()):
    """
    Trains a fresh agent in a fresh GridWorld(**env_params).

    record_episodes: episode indices whose full trajectory is kept in history["trajectories"]

    Returns (agent, history). history holds per-episode arrays over all n_episodes:
    "length", "total_reward", "food_left" (at episode end), "action_counts" ({category: array}),
    plus "trajectories" ({episode index: trajectory dict}).
    """
    env = GridWorld(**env_params, seed=seed)
    agent = QLearningAgent(n_actions=env.n_actions, **agent_params, seed=seed)
    record_episodes = set(record_episodes)

    lengths, total_rewards, food_left = [], [], []
    counts = {cat: [] for cat in ACTION_CATS}
    trajectories = {}
    for episode in range(n_episodes):
        traj = run_episode(env, agent, learn=True)
        agent.decay_epsilon() # decay epsilon after each episode
        lengths.append(len(traj["reward"])) # a reward for each step; so len(reward) = total steps
        total_rewards.append(traj["reward"].sum())
        food_left.append(traj["food_left"][-1])
        for cat, count in action_counts(traj).items():
            counts[cat].append(count)
        if episode in record_episodes:
            trajectories[episode] = traj

    history = {
        "length": np.array(lengths),
        "total_reward": np.array(total_rewards),
        "food_left": np.array(food_left),
        "action_counts": {cat: np.array(c) for cat, c in counts.items()},
        "trajectories": trajectories,
    }
    return agent, history


def evaluate(agent, env_params, n_episodes=1, seed=None):
    """
    Frozen policy: greedy actions (epsilon = 0) and no Q-table updates.
    A fixed seed gives the same sequence of food layouts / start positions to every agent.
    Returns a list of trajectory dicts from run_episode.
    """
    eval_env = GridWorld(**env_params, seed=seed)
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0
    trajs = [run_episode(eval_env, agent, learn=False) for _ in range(n_episodes)]
    agent.epsilon = saved_epsilon
    return trajs


def is_trained(agent, state):
    """
    True if the agent has Q-values for this state from training. Unseen states (and states only ever
    reached as an episode's final state) still have all-zero Q-values.
    """
    return state in agent.q_table and np.any(agent.q_table[state] != 0)


def summarize(trajs, agent, env_params):
    """
    Summary statistics over a list of evaluation trajectories.
    """
    target = env_params["glucose_target"]
    lengths = np.array([len(t["reward"]) for t in trajs])
    eaten = np.array([action_counts(t)["eat"] for t in trajs])
    # glucose[1:] is glucose after each step; glucose[:-1] is glucose when each action was chosen
    above = np.concatenate([t["glucose"][1:] > target for t in trajs])
    eat_glucose = np.concatenate([t["glucose"][:-1][t["categories"] == "eat"] for t in trajs])
    decision_states = [s for t in trajs for s in t["states"][:-1]] # states where an action was chosen
    return {
        "steps": lengths.mean(),
        "survived": np.mean([t["glucose"][-1] > 0 for t in trajs]), # reached max_steps without starving
        "food eaten": eaten.mean(),
        "share eaten": (eaten / env_params["n_food"]).mean(),
        "steps above target": above.mean(),
        "eats above target": (eat_glucose >= target).mean() if len(eat_glucose) else np.nan,
        "unseen": np.mean([not is_trained(agent, s) for s in decision_states]),
    }


def print_summary_table(summaries):
    """
    summaries: dict of {(agent label, env label): dict from summarize}
    """
    stat_names = list(next(iter(summaries.values())))
    print(f"{'agent':<18}{'eval env':<10}" + "".join(f"{s:>20}" for s in stat_names))
    for (a_label, e_label), stats in summaries.items():
        print(f"{a_label:<18}{e_label:<10}" + "".join(f"{stats[s]:>20.2f}" for s in stat_names))


# ----------------------------------------------------------
# Plots across training
# ----------------------------------------------------------

def moving_average(x, window):
    """
    Mean over a sliding window; output[i] averages x[i : i + window], so it has len(x) - window + 1 entries.
    """
    return np.convolve(x, np.ones(window) / window, mode="valid")


def _mark_episodes(ax, episodes):
    for ep in episodes:
        ax.axvline(ep, color="gray", linestyle=":", linewidth=1)


def plot_episode_lengths(history, window, mark_episodes=(), fname=None):
    """
    Steps survived per episode (light) and its moving average (dark); dotted lines at mark_episodes.
    """
    lengths = history["length"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lengths, color=LIGHT_LINE_COLOR, linewidth=1, label="per episode")
    ax.plot(np.arange(window - 1, len(lengths)), moving_average(lengths, window), color=LINE_COLOR,
            linewidth=2, label=f"{window}-episode moving average")
    _mark_episodes(ax, mark_episodes)
    ax.set_xlabel("episode")
    ax.set_ylabel("steps survived")
    ax.set_title("Episode length")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300)
    plt.show()


def plot_action_mix(history, n_food, window, mark_episodes=(), fname=None):
    """
    Left: moving average of each episode's share of steps per action category.
    Right: food left at episode end, per episode and moving average.
    """
    n_episodes = len(history["length"])
    ma_x = np.arange(window - 1, n_episodes) # episode index at the end of each window
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for cat in ACTION_CATS:
        share = history["action_counts"][cat] / history["length"]
        ax.plot(ma_x, moving_average(share, window), color=CAT_COLORS[cat], linewidth=2, label=cat)
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"share of steps\n(mean over {window} episodes)")
    ax.set_title(f"Action mix, {window}-episode moving average")
    ax.legend(frameon=False, loc="upper left", ncol=2)

    ax = axes[1]
    food_left = history["food_left"]
    ax.plot(food_left, color=LIGHT_LINE_COLOR, linewidth=0.8, alpha=0.6, label="per episode")
    ax.plot(ma_x, moving_average(food_left, window), color=LINE_COLOR, linewidth=2,
            label=f"{window}-episode moving average")
    ax.set_ylim(0, n_food)
    ax.set_ylabel("food left at end")
    ax.set_title("Food left over")
    ax.legend(frameon=False)

    for ax in axes:
        _mark_episodes(ax, mark_episodes)
        ax.set_xlabel("episode")
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300)
    plt.show()


# ----------------------------------------------------------
# Plots of individual episodes
# ----------------------------------------------------------

def episode_title(label, traj):
    """
    Multi-line panel title: label, epsilon, length, food eaten, idle count.
    """
    counts = action_counts(traj)
    return (f"{label}\neps={traj['epsilon']:.2f}, {len(traj['reward'])} steps\n"
            f"ate {counts['eat']}/{len(traj['food_start'])} · idle {counts['idle']}")


def plot_trajectories(trajs, env_params, titles=None, color=LINE_COLOR, fname=None):
    """
    One column per episode; rows: glucose, action strip, food left, reward, cumulative reward.

    trajs: dict of {label: trajectory dict from run_episode}
    env_params: GridWorld parameters, for the glucose target/limit and food-count axis
    """
    labels = list(trajs.keys())
    n = len(labels)
    fig, axes = plt.subplots(5, n, figsize=(3.4 * n, 11.5), sharex="col", sharey="row",
                             squeeze=False, gridspec_kw={"height_ratios": [3, 0.8, 2, 2, 2]})
    # fixed margins instead of tight/constrained layout, which collapse this grid when saving at high dpi
    fig.subplots_adjust(left=0.06, right=0.99, top=0.91, bottom=0.05, hspace=0.25, wspace=0.12)

    for col, label in enumerate(labels):
        traj = trajs[label]
        steps = np.arange(1, len(traj["reward"]) + 1)

        # glucose (index 0 = start of episode, before any step)
        ax = axes[0, col]
        ax.plot(np.arange(len(traj["glucose"])), traj["glucose"], color=color, linewidth=2)
        ax.axhline(env_params["glucose_target"], color="gray", linestyle="--", linewidth=1)
        eat_steps = steps[traj["categories"] == "eat"]
        ax.plot(eat_steps, np.zeros_like(eat_steps), "|", color=CAT_COLORS["eat"], markersize=8,
                clip_on=False)
        ax.set_ylim(0, env_params["glucose_max"])
        ax.set_title(titles[col] if titles else str(label), fontsize=10)

        # action strip: one row of ticks per category
        ax = axes[1, col]
        for i, cat in enumerate(STRIP_CATS):
            cat_steps = steps[traj["categories"] == cat]
            ax.plot(cat_steps, np.full(len(cat_steps), i), "|", color=CAT_COLORS[cat],
                    markersize=7, markeredgewidth=1.5)
        ax.set_ylim(len(STRIP_CATS) - 0.4, -0.6) # first category on top
        ax.set_yticks(range(len(STRIP_CATS)))
        ax.set_yticklabels(STRIP_CATS)
        ax.tick_params(axis="y", length=0)

        # food remaining (index 0 = start of episode)
        ax = axes[2, col]
        ax.plot(np.arange(len(traj["food_left"])), traj["food_left"], color=color, linewidth=2,
                drawstyle="steps-post")
        ax.set_ylim(0, env_params["n_food"])

        # per-step reward
        ax = axes[3, col]
        ax.plot(steps, traj["reward"], color=color, linewidth=1.5)
        ax.axhline(0, color="gray", linewidth=0.8)

        # cumulative reward
        ax = axes[4, col]
        ax.plot(steps, traj["cum_reward"], color=color, linewidth=2)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("step")

    axes[0, 0].set_ylabel("glucose")
    axes[1, 0].set_ylabel("action")
    axes[2, 0].set_ylabel("food left")
    axes[3, 0].set_ylabel("reward")
    axes[4, 0].set_ylabel("cumulative reward")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1]:
        ax.spines["left"].set_visible(False)
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight") # trim leftover whitespace
    plt.show()


def plot_episode_maps(trajs, env_params, titles=None, mode="path", jitter=0.12, fname=None, seed=0):
    """
    One grid map per episode, with food eaten (filled) / left over (hollow).

    trajs: dict of {label: trajectory dict from run_episode}
    env_params: GridWorld parameters, for the grid size
    mode: "path" (route colored by step) or "heatmap" (steps spent on each tile)
    """
    labels = list(trajs.keys())
    n = len(labels)
    g = env_params["grid_size"]
    rng = np.random.default_rng(seed) # jitter only; fixed so reruns look the same
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4.2), squeeze=False)
    axes = axes[0]

    # shared across panels so a color means the same step everywhere;
    # spans the longest plotted episode (not max_steps) so short episodes aren't all pale
    step_norm = Normalize(1, max(len(trajs[label]["reward"]) for label in labels))
    if mode == "heatmap":
        visits = {}
        for label in labels:
            v = np.zeros((g, g))
            for r, c in trajs[label]["positions"]:
                v[r, c] += 1
            visits[label] = v
        visit_norm = Normalize(0, max(v.max() for v in visits.values()))

    for i, (ax, label) in enumerate(zip(axes, labels)):
        traj = trajs[label]
        pos = traj["positions"]
        steps = np.arange(1, len(pos))

        if mode == "heatmap":
            mappable = ax.imshow(visits[label], cmap="Blues", norm=visit_norm,
                                 extent=(-0.5, g - 0.5, g - 0.5, -0.5))
        else:
            # plot (x, y) = (col, row)
            xy = pos[:, ::-1].astype(float) + rng.uniform(-jitter, jitter, size=pos.shape)
            segments = np.stack([xy[:-1], xy[1:]], axis=1)
            mappable = LineCollection(segments, cmap=STEP_CMAP, norm=step_norm, linewidth=1.5)
            mappable.set_array(steps)
            ax.add_collection(mappable)
            stayed = np.all(pos[1:] == pos[:-1], axis=1)
            ax.scatter(xy[1:][stayed, 0], xy[1:][stayed, 1], c=steps[stayed], cmap=STEP_CMAP,
                       norm=step_norm, s=14, zorder=3)
            ax.scatter(*xy[0], marker="o", s=70, facecolor="white", edgecolor="black",
                       linewidth=1.5, zorder=5)
            ax.scatter(*xy[-1], marker="X", s=80, facecolor="black", edgecolor="white",
                       linewidth=1, zorder=5)

        # food: filled = eaten, hollow = left over
        for r, c in traj["food_start"]:
            if (r, c) in traj["eaten_at"]:
                face = (STEP_CMAP(step_norm(traj["eaten_at"][(r, c)])) if mode == "path"
                        else CAT_COLORS["eat"])
                ax.scatter(c, r, marker="s", s=110, facecolor=face, edgecolor=CAT_COLORS["eat"],
                           linewidth=2, zorder=4)
            else:
                ax.scatter(c, r, marker="s", s=110, facecolor="none", edgecolor=FOOD_LEFT_COLOR,
                           linewidth=1.5, zorder=4)

        # grid lines between tiles
        ax.set_xticks(np.arange(-0.5, g), minor=True)
        ax.set_yticks(np.arange(-0.5, g), minor=True)
        ax.grid(which="minor", color="#e0dfda", linewidth=0.8)
        ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        ax.set_xlim(-0.5, g - 0.5)
        ax.set_ylim(g - 0.5, -0.5) # row 0 at the top, same as the grid array
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(False)
        n_eaten = len(traj["eaten_at"])
        default_title = (f"{label}\nate {n_eaten}/{len(traj['food_start'])}, "
                         f"{len(traj['reward'])} steps")
        ax.set_title(titles[i] if titles else default_title)

    handles = [
        Line2D([], [], marker="s", linestyle="", markersize=9, markerfacecolor=STEP_CMAP(0.5) if mode == "path"
               else CAT_COLORS["eat"], markeredgecolor=CAT_COLORS["eat"], markeredgewidth=2, label="food eaten"),
        Line2D([], [], marker="s", linestyle="", markersize=9, markerfacecolor="none",
               markeredgecolor=FOOD_LEFT_COLOR, markeredgewidth=1.5, label="food left over"),
    ]
    if mode == "path":
        handles += [
            Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor="white",
                   markeredgecolor="black", markeredgewidth=1.5, label="start"),
            Line2D([], [], marker="X", linestyle="", markersize=9, markerfacecolor="black",
                   markeredgecolor="white", label="end"),
        ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    plt.tight_layout(rect=(0, 0.08, 0.93, 1))
    cax = fig.add_axes((0.94, 0.2, 0.008, 0.6))
    fig.colorbar(mappable, cax=cax, label="step" if mode == "path" else "steps on tile")
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight") # keeps multi-line titles from being cut off
    plt.show()


# ----------------------------------------------------------
# Plots comparing agents across environments
# ----------------------------------------------------------

# one number per evaluation episode; fn(trajectory, env params)
PER_EPISODE_METRICS = {
    "steps survived": lambda t, p: len(t["reward"]),
    "food eaten": lambda t, p: action_counts(t)["eat"],
    "mean |glucose - target|": lambda t, p: np.abs(t["glucose"] - p["glucose_target"]).mean(),
}


def plot_condition_outcomes(evals, eval_env_params, agent_colors, fname=None):
    """
    Box plots of per-episode outcomes for every (agent, eval env) condition,
    grouped by eval env and colored by agent.

    evals: dict of {(agent label, env label): list of trajectories}
    eval_env_params: dict of {env label: GridWorld parameters}
    agent_colors: dict of {agent label: color}; also sets the agent order
    """
    env_labels = list(eval_env_params)
    n_agents = len(agent_colors)
    width = 0.7 / n_agents

    fig, axes = plt.subplots(1, len(PER_EPISODE_METRICS), figsize=(4.2 * len(PER_EPISODE_METRICS), 4))
    for ax, (metric, fn) in zip(axes, PER_EPISODE_METRICS.items()):
        for j, (a_label, color) in enumerate(agent_colors.items()):
            data = [[fn(t, eval_env_params[e_label]) for t in evals[(a_label, e_label)]]
                    for e_label in env_labels]
            positions = np.arange(len(env_labels)) + (j - (n_agents - 1) / 2) * width
            ax.boxplot(data, positions=positions, widths=width * 0.85, patch_artist=True,
                       boxprops=dict(facecolor=color, edgecolor=color, alpha=0.35),
                       medianprops=dict(color=color, linewidth=2),
                       whiskerprops=dict(color=color), capprops=dict(color=color),
                       flierprops=dict(marker="o", markersize=3, markerfacecolor=color,
                                       markeredgecolor="none", alpha=0.5))
        ax.set_xticks(range(len(env_labels)))
        ax.set_xticklabels([f"{e} env\n(n_food={eval_env_params[e]['n_food']})" for e in env_labels])
        ax.set_title(metric)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_xlabel("evaluation environment")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.6, edgecolor=c) for c in agent_colors.values()]
    fig.legend(handles, list(agent_colors), loc="lower center", ncol=n_agents, frameon=False)
    plt.tight_layout(rect=(0, 0.07, 1, 1))
    if fname:
        plt.savefig(fname, dpi=300)
    plt.show()


def running_mean_glucose(trajs, min_running=0.1):
    """
    Mean glucose at each step over the episodes still running at that step;
    NaN once fewer than min_running of the episodes are still running.
    """
    max_len = max(len(t["glucose"]) for t in trajs)
    padded = np.full((len(trajs), max_len), np.nan) # NaN after an episode ends
    for i, t in enumerate(trajs):
        padded[i, :len(t["glucose"])] = t["glucose"]
    running = np.mean(~np.isnan(padded), axis=0)
    mean_glucose = np.nanmean(padded, axis=0)
    mean_glucose[running < min_running] = np.nan
    return mean_glucose


def plot_mean_glucose(evals,eval_env_params, agent_colors, min_running=0.1, fname=None):
    """
    Mean glucose per step over the episodes still running at that step; one panel per eval env,
    one line per agent. Lines stop once fewer than min_running of the episodes are still running.
    """
    fig, axes = plt.subplots(1, len(eval_env_params), figsize=(6 * len(eval_env_params), 4),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (e_label, params) in zip(axes, eval_env_params.items()):
        for a_label, color in agent_colors.items():
            mean_glucose = running_mean_glucose(evals[(a_label, e_label)], min_running)
            ax.plot(mean_glucose, color=color, linewidth=2, label=a_label)
        ax.axhline(params["glucose_target"], color="gray", linestyle="--", linewidth=1)
        ax.set_ylim(0, params["glucose_max"])
        ax.set_title(f"{e_label} env (n_food={params['n_food']})")
        ax.set_xlabel("step")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("mean glucose (running episodes)")
    axes[0].legend(frameon=False)
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300)
    plt.show()


# ----------------------------------------------------------
# Variability across agents
# ----------------------------------------------------------
# These take several independently trained agents per condition (one per seed), so the spread
# they show is between agents, not just between episodes of a single agent.

def train_agents(env_params, agent_params, n_episodes, seeds):
    """
    Trains one fresh agent per seed with train_agent (no recorded trajectories, to save memory).
    Returns (agents, histories), both lists in seed order.
    """
    agents, histories = [], []
    for seed in seeds:
        agent, history = train_agent(env_params, agent_params, n_episodes, seed=seed)
        agents.append(agent)
        histories.append(history)
    return agents, histories


def evaluate_agents(agents, env_params, n_episodes, seed):
    """
    evaluate() for every agent with the same seed, so all agents face the same food layouts.
    Returns a list (one entry per agent) of lists of trajectories.
    """
    return [evaluate(agent, env_params, n_episodes=n_episodes, seed=seed) for agent in agents]


def episode_metric_table(agent_evals, env_params):
    """
    {metric: array of shape (n_agents, n_episodes)} for every metric in PER_EPISODE_METRICS.
    agent_evals: list over agents of lists of trajectories (from evaluate_agents)
    """
    return {metric: np.array([[fn(t, env_params) for t in trajs] for trajs in agent_evals])
            for metric, fn in PER_EPISODE_METRICS.items()}


def variance_partition(agent_evals, env_params):
    """
    Share of each metric's per-episode variance that comes from differences between agents:
    var(agent means) / (var(agent means) + mean within-agent var). With the same number of episodes
    per agent this is the law of total variance, so the two parts add up to the total variance.
    Noise in each agent's mean inflates the between-agent part by about (within var) / n_episodes,
    which is small for a few hundred episodes. NaN if the metric doesn't vary at all.
    """
    shares = {}
    for metric, values in episode_metric_table(agent_evals, env_params).items():
        between = values.mean(axis=1).var()
        within = values.var(axis=1).mean()
        total = between + within
        shares[metric] = between / total if total > 0 else np.nan
    return shares


# per-episode training curves; fn(history) -> one value per episode
LEARNING_CURVES = {
    "steps survived": lambda h: h["length"],
    "food left at end": lambda h: h["food_left"],
    "share of steps: eat": lambda h: h["action_counts"]["eat"] / h["length"],
    "share of steps: idle": lambda h: h["action_counts"]["idle"] / h["length"],
}


def plot_learning_curves_across_agents(groups, colors, window, fname=None):
    """
    Moving-average training curves: one thin line per agent, plus the mean across agents (bold)
    with a band of ± 1 SD across agents (not across episodes).

    groups: dict of {group label: list of histories}
    colors: dict of {group label: color}
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for ax, (metric, fn) in zip(axes.flat, LEARNING_CURVES.items()):
        for label, histories in groups.items():
            curves = np.array([moving_average(fn(h), window) for h in histories]) # (agents, episodes)
            x = np.arange(window - 1, window - 1 + curves.shape[1])
            for curve in curves:
                ax.plot(x, curve, color=colors[label], linewidth=0.6, alpha=0.35)
            mean, sd = curves.mean(axis=0), curves.std(axis=0)
            ax.fill_between(x, mean - sd, mean + sd, color=colors[label], alpha=0.12, linewidth=0)
            ax.plot(x, mean, color=colors[label], linewidth=2,
                    label=f"{label} (mean of {len(histories)} agents ± 1 SD)")
        ax.set_title(metric)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("episode")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"{window}-episode moving average")
    axes[0, 0].legend(frameon=False)
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300)
    plt.show()


def plot_per_agent_outcomes(agent_evals, env_params, color, title, agent_labels=None, fname=None):
    """
    One box per agent: its distribution over evaluation episodes. Agents are sorted by their mean
    separately in each panel (tick labels say which agent is which); diamond = agent mean,
    dashed line = mean over all agents. Panel titles give the between-agent share of variance.

    agent_evals: list over agents of lists of trajectories (from evaluate_agents)
    agent_labels: tick label per agent (e.g. its seed); defaults to 0..n_agents-1
    """
    table = episode_metric_table(agent_evals, env_params)
    shares = variance_partition(agent_evals, env_params)
    n = len(agent_evals)
    agent_labels = list(agent_labels) if agent_labels is not None else list(range(n))

    fig, axes = plt.subplots(1, len(table), figsize=((0.4 * n + 1.8) * len(table), 4.2))
    for ax, (metric, values) in zip(axes, table.items()):
        order = np.argsort(values.mean(axis=1))
        ax.boxplot([values[i] for i in order], positions=np.arange(n), widths=0.6, patch_artist=True,
                   boxprops=dict(facecolor=color, edgecolor=color, alpha=0.35),
                   medianprops=dict(color=color, linewidth=2),
                   whiskerprops=dict(color=color), capprops=dict(color=color),
                   flierprops=dict(marker="o", markersize=3, markerfacecolor=color,
                                   markeredgecolor="none", alpha=0.5))
        ax.scatter(np.arange(n), values[order].mean(axis=1), marker="D", s=24, color="black", zorder=3)
        ax.axhline(values.mean(), color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(np.arange(n))
        ax.set_xticklabels([agent_labels[i] for i in order])
        share = "n/a" if np.isnan(shares[metric]) else f"{shares[metric]:.0%}"
        ax.set_title(f"{metric}\nbetween-agent share of variance: {share}", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_xlabel("agent (sorted by mean)")
    fig.suptitle(title)
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300)
    plt.show()


def plot_variance_partition(partitions, colors, fname=None):
    """
    One panel per metric, one bar per condition: share of the per-episode variance that comes
    from differences between agents (see variance_partition).

    partitions: dict of {condition label: dict from variance_partition}
    colors: dict of {condition label: color}
    """
    conditions = list(partitions)
    y = np.arange(len(conditions))
    fig, axes = plt.subplots(1, len(PER_EPISODE_METRICS), figsize=(4.2 * len(PER_EPISODE_METRICS),
                             0.55 * len(conditions) + 1.6), sharey=True)
    for ax, metric in zip(axes, PER_EPISODE_METRICS):
        shares = np.array([partitions[c][metric] for c in conditions])
        ax.barh(y, np.nan_to_num(shares), height=0.6, color=[colors[c] for c in conditions])
        for yi, share in zip(y, shares):
            ax.text(0.02 if np.isnan(share) else share + 0.02, yi, "n/a" if np.isnan(share) else f"{share:.0%}",
                    va="center", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_title(metric)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(conditions)
    axes[0].invert_yaxis() # first condition on top (shared y, so this flips every panel)
    axes[0].set_xlabel("between-agent share of variance")
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300)
    plt.show()


def plot_transfer_slopes(evals, eval_env_params, agent_colors, fname=None):
    """
    Each thin line is one agent: its mean over evaluation episodes in each eval env, colored by the
    environment it was trained in. Bold line = mean across that training condition's agents.
    Training conditions are nudged apart horizontally so their lines don't hide each other.

    evals: dict of {(training label, env label): list over agents of lists of trajectories}
    eval_env_params: dict of {env label: GridWorld parameters}
    agent_colors: dict of {training label: color}; also sets the order
    """
    env_labels = list(eval_env_params)
    n_groups = len(agent_colors)
    fig, axes = plt.subplots(1, len(PER_EPISODE_METRICS), figsize=(4.2 * len(PER_EPISODE_METRICS), 4))
    for ax, (metric, fn) in zip(axes, PER_EPISODE_METRICS.items()):
        for j, (a_label, color) in enumerate(agent_colors.items()):
            # (n_agents, n_envs): each agent's mean in each eval env
            means = np.array([[np.mean([fn(t, eval_env_params[e]) for t in trajs])
                               for trajs in evals[(a_label, e)]] for e in env_labels]).T
            x = np.arange(len(env_labels)) + (j - (n_groups - 1) / 2) * 0.08
            for row in means:
                ax.plot(x, row, color=color, linewidth=0.8, alpha=0.45, marker="o", markersize=3)
            ax.plot(x, means.mean(axis=0), color=color, linewidth=2.5, marker="o", markersize=8,
                    label=f"{a_label} (mean of {len(means)})")
        ax.set_xticks(range(len(env_labels)))
        ax.set_xticklabels([f"{e} env\n(n_food={eval_env_params[e]['n_food']})" for e in env_labels])
        ax.set_xlim(-0.4, len(env_labels) - 0.6)
        ax.set_title(metric)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_xlabel("evaluation environment")
    axes[0].set_ylabel("agent mean over eval episodes")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n_groups, frameon=False)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    if fname:
        plt.savefig(fname, dpi=300)
    plt.show()


def plot_mean_glucose_across_agents(evals, eval_env_params, agent_colors, min_running=0.1, fname=None):
    """
    Like plot_mean_glucose, but with one thin line per agent (its mean over its own eval episodes)
    and a bold line pooling every agent's episodes. One panel per eval env.

    evals: dict of {(training label, env label): list over agents of lists of trajectories}
    """
    fig, axes = plt.subplots(1, len(eval_env_params), figsize=(6 * len(eval_env_params), 4),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (e_label, params) in zip(axes, eval_env_params.items()):
        for a_label, color in agent_colors.items():
            agent_evals = evals[(a_label, e_label)]
            for trajs in agent_evals:
                ax.plot(running_mean_glucose(trajs, min_running), color=color, linewidth=0.7, alpha=0.4)
            pooled = [t for trajs in agent_evals for t in trajs]
            ax.plot(running_mean_glucose(pooled, min_running), color=color, linewidth=2.5,
                    label=f"{a_label} (all {len(agent_evals)} agents)")
        ax.axhline(params["glucose_target"], color="gray", linestyle="--", linewidth=1)
        ax.set_ylim(0, params["glucose_max"])
        ax.set_title(f"{e_label} env (n_food={params['n_food']})")
        ax.set_xlabel("step")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("mean glucose (running episodes)")
    axes[0].legend(frameon=False)
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300)
    plt.show()


def eat_advantage_by_glucose(agent, glucose_max):
    """
    For the state of standing on food, (0, 'O', glucose bin), at each glucose bin:
    Q(EAT) minus the best other action's Q-value. > 0 means the greedy policy eats at that glucose.
    NaN for bins the agent has no trained Q-values for.
    Returns (bins, advantages); bin b covers glucose 10b to 10b + 9 (see discretize_state).
    """
    bins = np.arange(glucose_max // 10 + 1)
    advantages = np.full(len(bins), np.nan)
    for b in bins:
        state = (0, "O", int(b))
        if is_trained(agent, state): # checks membership first, so it doesn't add the state to the defaultdict
            q = agent.q_table[state]
            advantages[b] = q[Action.EAT] - max(q[a] for a in Action if a != Action.EAT)
    return bins, advantages


def plot_eat_policy(groups, colors, glucose_target, glucose_max, fname=None):
    """
    How each agent's greedy policy decides to eat when standing on food, as a function of glucose.
    Left: Q(EAT) - best other Q, one line per agent (> 0 = eats). Right: share of agents whose greedy
    action is EAT, among agents with trained Q-values in that bin. Dashed line = glucose target.

    groups: dict of {group label: list of agents}
    colors: dict of {group label: color}
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, agents in groups.items():
        advantages = []
        for agent in agents:
            bins, adv = eat_advantage_by_glucose(agent, glucose_max)
            advantages.append(adv)
            axes[0].plot(bins * 10, adv, color=colors[label], linewidth=0.8, alpha=0.5, marker="o", markersize=3)
        advantages = np.array(advantages)
        trained = ~np.isnan(advantages)
        n_trained = trained.sum(axis=0)
        eats = np.where(trained, advantages > 0, False).sum(axis=0)
        share = np.where(n_trained > 0, eats / np.maximum(n_trained, 1), np.nan)
        axes[1].plot(bins * 10, share, color=colors[label], linewidth=2, marker="o", markersize=7,
                     label=f"{label} ({len(agents)} agents)")

    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].set_ylabel("Q(EAT) - best other Q")
    axes[0].set_title("Eating preference on a food tile, per agent")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_ylabel("share of agents that eat")
    axes[1].set_title("Agents whose greedy action is EAT")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.axvline(glucose_target, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("glucose (start of 10-unit bin)")
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300)
    plt.show()


def greedy_action(agent, state):
    """
    Highest-Q action allowed in this state (EAT only when standing on food); ties go to the lowest action.
    """
    q = agent.q_table[state]
    valid = [a for a in Action if a != Action.EAT or state[1] == "O"]
    return max(valid, key=lambda a: q[a])


def policy_agreement(agents):
    """
    (agreement, n_shared): n_agents x n_agents matrices. agreement[i, j] is the share of states with
    trained Q-values in both agents i and j where their greedy actions match; n_shared[i, j] is the
    number of those states.
    """
    trained = [{s for s in agent.q_table if is_trained(agent, s)} for agent in agents]
    policies = [{s: greedy_action(agent, s) for s in states} for agent, states in zip(agents, trained)]
    n = len(agents)
    agreement, n_shared = np.full((n, n), np.nan), np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            shared = trained[i] & trained[j]
            n_shared[i, j] = len(shared)
            if shared:
                agreement[i, j] = np.mean([policies[i][s] == policies[j][s] for s in shared])
    return agreement, n_shared


def plot_policy_agreement(groups, fname=None):
    """
    Heatmap of policy_agreement over every agent, stacked in group order, with white lines between groups.
    Returns {(group a, group b): mean agreement over pairs of distinct agents}.

    groups: dict of {group label: list of agents}
    """
    agents = [agent for group in groups.values() for agent in group]
    agreement, _ = policy_agreement(agents)
    sizes = [len(group) for group in groups.values()]
    edges = np.concatenate([[0], np.cumsum(sizes)])

    block_means = {}
    labels = list(groups)
    off_diagonal = ~np.eye(len(agents), dtype=bool)
    for a in range(len(labels)):
        for b in range(len(labels)):
            block = agreement[edges[a]:edges[a + 1], edges[b]:edges[b + 1]]
            mask = off_diagonal[edges[a]:edges[a + 1], edges[b]:edges[b + 1]]
            block_means[(labels[a], labels[b])] = np.nanmean(block[mask]) if mask.any() else np.nan

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(agreement, cmap=STEP_CMAP, vmin=0, vmax=1)
    for edge in edges[1:-1]:
        ax.axhline(edge - 0.5, color="white", linewidth=3)
        ax.axvline(edge - 0.5, color="white", linewidth=3)
    centers = (edges[:-1] + edges[1:]) / 2 - 0.5
    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    ax.set_yticks(centers)
    ax.set_yticklabels(labels, rotation=90, va="center")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Greedy-action agreement between agents\n(over states both agents trained on)")
    fig.colorbar(im, ax=ax, label="share of shared states with the same greedy action")
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300)
    plt.show()
    return block_means
