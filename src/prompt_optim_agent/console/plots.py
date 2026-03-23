"""Matplotlib plot generation for MCTS results."""

import os
from collections import deque


def _try_import_matplotlib():
    """Import matplotlib in non-interactive mode, or return None if unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


class PlotManager:
    """Generates and saves MCTS visualization plots."""

    def __init__(self):
        self._reward_history: list[dict] = []
        self._plt = _try_import_matplotlib()

    def _plots_dir(self, log_dir: str) -> str:
        path = os.path.join(log_dir, "plots")
        os.makedirs(path, exist_ok=True)
        return path

    def _save(self, fig, log_dir: str, name: str) -> None:
        fig.tight_layout()
        fig.savefig(os.path.join(self._plots_dir(log_dir), name), dpi=150)
        self._plt.close(fig)

    # ------------------------------------------------------------------
    # Incremental reward progress
    # ------------------------------------------------------------------
    def update_reward_plot(
        self, iteration: int, node_rewards: list[float], best_reward: float, log_dir: str,
    ) -> None:
        self._reward_history.append({
            "iteration": iteration,
            "best": best_reward,
            "mean": sum(node_rewards) / max(len(node_rewards), 1),
            "min": min(node_rewards) if node_rewards else 0,
        })
        if self._plt is None:
            return
        try:
            iters = [h["iteration"] for h in self._reward_history]
            fig, ax = self._plt.subplots(figsize=(8, 5))
            for key, style, color in [
                ("best", "o-", "#2ecc71"),
                ("mean", "s--", "#3498db"),
                ("min", "^:", "#e74c3c"),
            ]:
                values = [h[key] for h in self._reward_history]
                ax.plot(iters, values, style, label=key.title(), color=color)
            ax.set(xlabel="Iteration", ylabel="Reward", title="MCTS Reward Progress")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(iters)
            self._save(fig, log_dir, "reward_progress.png")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Tree visualization
    # ------------------------------------------------------------------
    def generate_tree(
        self, nodes: list[dict], log_dir: str, selected_node_id: int = -1,
    ) -> None:
        if self._plt is None or not nodes:
            return
        try:
            self._generate_tree_impl(nodes, log_dir, selected_node_id)
        except Exception:
            pass

    def _generate_tree_impl(
        self, nodes: list[dict], log_dir: str, selected_node_id: int,
    ) -> None:
        import matplotlib.patches as mpatches

        children_map: dict[int, list[int]] = {}
        node_map: dict[int, dict] = {}
        for n in nodes:
            node_map[n["id"]] = n
            parent_id = n.get("parent", -1)
            if parent_id >= 0:
                children_map.setdefault(parent_id, []).append(n["id"])

        positions = self._compute_tree_positions(node_map, children_map)

        fig, ax = self._plt.subplots(figsize=(10, 6))

        # Edges
        for parent_id, child_ids in children_map.items():
            if parent_id not in positions:
                continue
            px, py = positions[parent_id]
            for cid in child_ids:
                if cid in positions:
                    cx, cy = positions[cid]
                    ax.plot([px, cx], [py, cy], "-", color="#bdc3c7", linewidth=1, zorder=1)

        # Nodes
        for nid, (x, y) in positions.items():
            reward = node_map[nid]["reward"]
            color = self._plt.cm.RdYlGn(max(0, min(1, reward)))
            is_selected = nid == selected_node_id
            edge_color = "#e74c3c" if is_selected else "black"
            edge_width = 3.0 if is_selected else 0.5
            size = 500 if is_selected else 200 + reward * 300
            ax.scatter(
                x, y, s=size, c=[color],
                edgecolors=edge_color, linewidth=edge_width, zorder=2,
            )
            ax.annotate(
                f"{nid}\n{reward:.2f}", (x, y),
                ha="center", va="center", fontsize=7, fontweight="bold", zorder=3,
            )
            if is_selected:
                circle = self._plt.Circle(
                    (x, y), 0.035, fill=False,
                    edgecolor="#e74c3c", linewidth=2.5, linestyle="--", zorder=4,
                )
                ax.add_patch(circle)

        ax.set_title("MCTS Search Tree")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.1, 1.1)
        ax.axis("off")

        legend_handles = [
            mpatches.Patch(color=self._plt.cm.RdYlGn(v), label=label)
            for v, label in [(0.0, "Low reward"), (0.5, "Mid reward"), (1.0, "High reward")]
        ]
        legend_handles.append(
            mpatches.Patch(edgecolor="#e74c3c", facecolor="none", linewidth=2, label="Selected")
        )
        ax.legend(handles=legend_handles, loc="lower right", fontsize=8)
        self._save(fig, log_dir, "tree.png")

    @staticmethod
    def _compute_tree_positions(
        node_map: dict[int, dict],
        children_map: dict[int, list[int]],
    ) -> dict[int, tuple[float, float]]:
        """BFS over the tree to assign (x, y) positions by depth."""
        depth_nodes: dict[int, list[int]] = {}
        queue = deque([0])
        visited: set[int] = set()
        while queue:
            nid = queue.popleft()
            if nid in visited or nid not in node_map:
                continue
            visited.add(nid)
            depth_nodes.setdefault(node_map[nid]["depth"], []).append(nid)
            queue.extend(children_map.get(nid, []))

        max_depth = max(depth_nodes.keys(), default=0)
        positions: dict[int, tuple[float, float]] = {}
        for depth, nids in depth_nodes.items():
            for i, nid in enumerate(nids):
                x = (i + 0.5) / len(nids)
                y = 1.0 - depth / max(max_depth, 1)
                positions[nid] = (x, y)
        return positions

    # ------------------------------------------------------------------
    # Final plots (distribution + path comparison)
    # ------------------------------------------------------------------
    def generate_distribution(self, rewards: list[float], log_dir: str) -> None:
        if self._plt is None:
            return
        try:
            fig, ax = self._plt.subplots(figsize=(8, 5))
            ax.hist(rewards, bins=max(5, len(rewards) // 2), color="#3498db", edgecolor="white", alpha=0.8)
            ax.set(xlabel="Reward", ylabel="Count", title="Node Reward Distribution")
            ax.grid(True, alpha=0.3)
            self._save(fig, log_dir, "reward_distribution.png")
        except Exception:
            pass

    def generate_path_comparison(self, paths: list[list[dict]], log_dir: str) -> None:
        if self._plt is None or not paths:
            return
        try:
            fig, ax = self._plt.subplots(figsize=(8, 5))
            for i, path in enumerate(paths):
                rewards = [n["reward"] for n in path]
                ax.plot(range(len(rewards)), rewards, "o-", label=f"Path {i}", linewidth=1.5, markersize=6)
            ax.set(xlabel="Depth", ylabel="Reward", title="Reward by Path Depth")
            ax.legend()
            ax.grid(True, alpha=0.3)
            self._save(fig, log_dir, "path_comparison.png")
        except Exception:
            pass
