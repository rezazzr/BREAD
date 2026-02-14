"""BreadConsole -- Rich-powered terminal output for BREAD experiments."""

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .plots import PlotManager
from .styles import PHASE_COLORS, reward_style, truncate


def _format_metric(metric: Any) -> str:
    """Format a metric value: floats get 4-decimal styling, others pass through."""
    if isinstance(metric, float):
        return f"[{reward_style(metric)}]{metric:.4f}[/]"
    return str(metric)


class BreadConsole:
    """Structured, colored console output for BREAD experiments."""

    def __init__(self, enabled: bool = True):
        self._console = Console(highlight=False) if enabled else Console(quiet=True)
        self._plots = PlotManager()

    # ------------------------------------------------------------------
    # Config / startup
    # ------------------------------------------------------------------
    def config_panel(
        self, task_name: str, search_algo: str, base_model: str,
        optim_model: str, iterations: int, depth_limit: int, expand_width: int,
    ) -> None:
        lines = [
            f"[bold]BREAD[/bold] -- {task_name} / {search_algo}",
            f"base_model: [cyan]{base_model}[/cyan]  optim_model: [cyan]{optim_model}[/cyan]",
            f"iterations: {iterations}  depth: {depth_limit}  expand: {expand_width}",
        ]
        self._console.print()
        self._console.print(Panel("\n".join(lines), border_style="bright_blue", padding=(0, 2)))

    def init_reward(self, reward: float) -> None:
        self._console.print(f"\n  init reward: [{reward_style(reward)}]{reward:.4f}[/]")

    # ------------------------------------------------------------------
    # Iteration / phase
    # ------------------------------------------------------------------
    def iteration_header(self, i: int, total: int) -> None:
        self._console.print()
        self._console.print(Rule(f" ITERATION {i + 1}/{total} ", style="bold white"))
        self._console.print()

    def phase(self, name: str, detail: str = "") -> None:
        color = PHASE_COLORS.get(name, "white")
        tag = f"[{color}][{name}][/]"
        self._console.print(f"  {tag} {detail}" if detail else f"  {tag}")

    # ------------------------------------------------------------------
    # Forward / gradient / optimize
    # ------------------------------------------------------------------
    def forward_results(self, labels, preds, correct, acc: float) -> None:
        marks = "".join("[green]\u2713[/]" if c == 1 else "[red]\u2717[/]" for c in correct)
        self._console.print(f"  [blue][FORWARD][/] acc: [{reward_style(acc)}]{acc:.4f}[/]  {marks}")

    def gradient_summary(self, gradient: str) -> None:
        self._console.print(f'  [magenta][GRADIENT][/] "{truncate(gradient, 200)}"')

    def optimization_result(self, new_prompts: list[str]) -> None:
        for i, p in enumerate(new_prompts):
            self._console.print(f'  [green][OPTIMIZE][/] prompt {i + 1}: "{truncate(p, 150)}"')

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------
    def node_info(
        self, node_id: int, depth: int, reward: float,
        q: float = 0.0, uct: float = 0.0, test_metric: Any = None,
    ) -> None:
        style = reward_style(reward)
        parts = [f"node [bold]{node_id}[/]", f"depth {depth}", f"reward [{style}]{reward:.4f}[/]"]
        if q:
            parts.append(f"Q {q:.4f}")
        if uct:
            parts.append(f"UCT {uct:.4f}")
        if test_metric is not None:
            parts.append(f"test {test_metric}")
        self._console.print("  " + "  |  ".join(parts))

    def children_table(self, children: list[dict]) -> None:
        table = Table(show_header=True, header_style="bold", padding=(0, 1))
        table.add_column("ID", justify="center", style="bold")
        table.add_column("Depth", justify="center")
        table.add_column("Reward", justify="center")
        for c in children:
            table.add_row(
                str(c["id"]),
                str(c["depth"]),
                Text(f"{c['reward']:.4f}", style=reward_style(c["reward"])),
            )
        self._console.print(table)

    def path_table(self, path_index: int, path_ids: list[int], mean_reward: float, mean_q: float) -> None:
        style = reward_style(mean_reward)
        ids_str = " -> ".join(str(i) for i in path_ids)
        self._console.print(
            f"  path {path_index}: [{style}]{ids_str}[/]  "
            f"mean_reward: [{style}]{mean_reward:.4f}[/]  mean_Q: {mean_q:.4f}"
        )

    # ------------------------------------------------------------------
    # Simulation / backprop
    # ------------------------------------------------------------------
    def backprop_summary(self, path_ids: list[int], cum_rewards: list[float]) -> None:
        ids_str = "->".join(str(i) for i in path_ids)
        rewards_str = ", ".join(f"{r:.2f}" for r in cum_rewards)
        self._console.print(f"  [yellow bold][BACKPROP][/] {ids_str}: [{rewards_str}]")

    def early_stop(self, node_id: int, reward: float, threshold: float) -> None:
        self._console.print(
            f"  [cyan bold][SIMULATE][/] early stop at node {node_id} "
            f"(reward {reward:.4f} > threshold {threshold:.4f})"
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def eval_result(self, prompt: str, metric: Any) -> None:
        short = truncate(prompt, 80)
        self._console.print(f'  [blue bold][EVAL][/] {_format_metric(metric)}  "{short}"')

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def results_header(self) -> None:
        self._console.print()
        self._console.print(Rule(" RESULTS ", style="bold green"))
        self._console.print()

    def search_complete(self, best_reward: float, best_prompt: str, elapsed: str) -> None:
        self._console.print(f"  Best eval reward: [{reward_style(best_reward)}]{best_reward:.4f}[/]")
        self._console.print(Panel(
            truncate(best_prompt, 200), title="Best prompt", border_style="green", padding=(0, 1),
        ))
        self._console.print(f"  Execution time: [bold]{elapsed}[/]")

    def selected_node_detail(
        self, node_id: int, depth: int, reward: float, test_metric: Any, prompt: str,
    ) -> None:
        style = reward_style(reward)
        self._console.print()
        self._console.print(Rule(" SELECTED NODE ", style="bold bright_green"))

        line = f"  node [bold]{node_id}[/]  |  depth {depth}  |  reward [{style}]{reward:.4f}[/]"
        if test_metric is not None:
            line += f"  |  test {_format_metric(test_metric)}"
        self._console.print(line)

        self._console.print(Panel(prompt, title="Selected prompt", border_style="bright_green", padding=(0, 1)))

    def test_result(self, label: str, metric: Any) -> None:
        self._console.print(f"  {label}: {_format_metric(metric)}")

    def status(self, msg: str) -> None:
        self._console.print(f"  {msg}")

    # ------------------------------------------------------------------
    # Plots (delegated to PlotManager)
    # ------------------------------------------------------------------
    def update_reward_plot(
        self, iteration: int, node_rewards: list[float], best_reward: float, log_dir: str,
    ) -> None:
        self._plots.update_reward_plot(iteration, node_rewards, best_reward, log_dir)

    def generate_final_plots(
        self, nodes: list[dict], paths: list, log_dir: str, selected_node_id: int = -1,
    ) -> None:
        self._plots.generate_distribution([n["reward"] for n in nodes], log_dir)
        self._plots.generate_path_comparison(paths, log_dir)
        self._plots.generate_tree(nodes, log_dir, selected_node_id=selected_node_id)
        self._console.print(f"  Plots saved to: [dim]{log_dir}/plots/[/]")
