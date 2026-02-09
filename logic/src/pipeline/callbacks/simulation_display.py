"""
Unified simulation display for WSmart-Route using Rich and Plotext.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import plotext as plt
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

# METRICS is now imported inside methods where needed to avoid unused import warning


class SimulationDisplay:
    """
    Real-time terminal dashboard for simulation testing.

    Provides high-performance visualization of:
    1. Overall simulation progress (samples * days)
    2. Policy-specific performance metrics (Averages)
    3. Historical progress chart (e.g., Profit/Overflows per day)
    """

    def __init__(
        self,
        policies: List[str],
        n_samples: int,
        total_days: int,
        chart_metric: str = "profit",
        refresh_rate: int = 2,
        theme: str = "dark",
    ):
        self.policies = policies
        self.n_samples = n_samples
        self.total_days = total_days
        self.chart_metric = chart_metric
        self.refresh_rate = refresh_rate
        self.theme = theme

        # Rich Components
        # If sys.stdout is redirected by LoggerWriter, we use the original terminal
        # to ensure the dashboard is visible and not logged.
        from logic.src.utils.logging.logger_writer import LoggerWriter

        main_console_out = sys.stdout
        if isinstance(main_console_out, LoggerWriter):
            main_console_out = main_console_out.terminal

        self.console = Console(file=main_console_out, force_terminal=True, force_interactive=True)
        self.layout = Layout()
        self.live: Optional[Live] = None
        self.progress: Optional[Progress] = None

        # Task IDs
        self.overall_task_id: Optional[TaskID] = None
        self.policy_tasks: Dict[str, TaskID] = {}

        # Tracking state
        self.start_time = time.time()
        self.colors = ["cyan", "magenta", "green", "yellow", "red", "blue", "white"]

        from logic.src.constants import SIM_METRICS

        self.policy_stats: Dict[str, Dict[str, Any]] = {
            pol: {"completed": 0, "metrics": {k: 0.0 for k in SIM_METRICS}} for pol in policies
        }

        # Data for chart: policy -> day -> list of values
        # We aggregate values from different samples for the same day
        self.daily_history: Dict[str, Dict[int, List[float]]] = {pol: defaultdict(list) for pol in policies}

        self._init_layout()

    def _init_layout(self) -> None:
        """Initialize the Rich layout."""
        self.layout.split(
            Layout(name="header", size=1),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=6),
        )
        self.layout["main"].split(
            Layout(name="chart", ratio=2),
            Layout(name="metrics", ratio=1),
        )

        # Initialize Progress
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None, complete_style="blue", finished_style="green"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
            expand=True,
        )

        # Overall Task
        total_steps = self.n_samples * len(self.policies) * self.total_days
        self.overall_task_id = self.progress.add_task("Overall Progress", total=total_steps)

        # Policy Tasks (optional, maybe too many if we have lots of policies)
        if len(self.policies) <= 10:
            for pol in self.policies:
                pol_total = self.n_samples * self.total_days
                self.policy_tasks[pol] = self.progress.add_task(f"Policy: {pol}", total=pol_total)

    def start(self) -> None:
        """Start the live display."""
        self.live = Live(
            self.layout,
            console=self.console,
            refresh_per_second=self.refresh_rate,
            screen=True,  # Isolate UI on alternate screen
            auto_refresh=True,
        )
        self.live.start()

    def stop(self) -> None:
        """Stop the live display."""
        if self.live:
            self.live.stop()

    def update(
        self,
        overall_completed: int,
        policy_updates: Dict[str, Dict[str, Any]],
        new_daily_data: List[Dict[str, Any]],
    ) -> None:
        """
        Update the display state from shared data.

        Args:
            overall_completed: Total simulation days finished across all workers.
            policy_updates: Dict mapping policy -> {'completed': int, 'metrics': dict}
            new_daily_data: List of dicts {'policy': str, 'day': int, 'metrics': dict}
        """
        # Update Progress
        if self.progress and self.overall_task_id is not None:
            self.progress.update(self.overall_task_id, completed=overall_completed)

        # Update Policy Stats and Tasks
        for pol, data in policy_updates.items():
            if pol in self.policy_stats:
                self.policy_stats[pol].update(data)
                if pol in self.policy_tasks and self.progress:
                    # Completed steps for this policy = samples_done * total_days + current_day_of_active_samples (simplified)
                    # For simplicity, we just use the progress reported by the worker counter if we can aggregate it.
                    # Here we just mark completed samples.
                    pol_completed = data.get("total_days_done", 0)
                    self.progress.update(self.policy_tasks[pol], completed=pol_completed)

        # Update History
        for entry in new_daily_data:
            pol = entry["policy"]
            day = entry["day"]
            metrics = entry["metrics"]
            if pol in self.daily_history and self.chart_metric in metrics:
                self.daily_history[pol][day].append(metrics[self.chart_metric])

        # Render and refresh Live
        if self.live:
            self.live.update(self._render_layout())

    def _render_layout(self) -> Layout:
        """Render the current state of the layout."""
        elapsed = time.time() - self.start_time
        header_text = Text(
            f" WSmart-Route Simulation • {time.strftime('%H:%M:%S', time.gmtime(elapsed))} • {len(self.policies)} Policies • {self.n_samples} Samples ",
            style="bold white on blue",
            justify="center",
        )
        self.layout["header"].update(header_text)

        self.layout["chart"].update(self._generate_chart())
        self.layout["metrics"].update(self._generate_metrics_table())

        if self.progress:
            self.layout["footer"].update(self.progress)

        return self.layout

    def _generate_chart(self) -> Panel:
        """Generate plotext chart showing average metrics per day."""
        plt.clf()
        plt.theme(self.theme)

        # Calculate plot dimensions for full-width layout
        # Be conservative to prevent line wrapping
        width = self.console.width - 6
        height = 12

        has_data = False
        for i, pol in enumerate(self.policies):
            history = self.daily_history[pol]
            if not history:
                continue

            days = sorted(history.keys())
            avg_values = [np.mean(history[d]) for d in days]

            if len(days) > 0:
                color = self.colors[i % len(self.colors)]
                plt.scatter(days, avg_values, color=color)  # Use scatter for points
                has_data = True

        if has_data:
            plt.title(f"Average {self.chart_metric.capitalize()} per Day")
            plt.xlabel("Day")
            # plt.ylabel(self.chart_metric.capitalize())
            # Hide legend to avoid overlapping mangling; policy is in header/footer
            pass
        else:
            plt.text("Waiting for simulation data...", x=0.5, y=0.5, alignment="center")

        plt.plotsize(max(width, 20), height)
        chart_text = Text.from_ansi(plt.build())
        return Panel(chart_text, title="📈 Performance History", border_style="blue", expand=True)

    def _generate_metrics_table(self) -> Panel:
        """Generate table showing aggregate metrics per policy."""
        from logic.src.constants import SIM_METRICS

        table = Table(show_header=True, header_style="bold magenta", expand=True, box=None, border_style="dim")
        table.add_column("Policy", style="cyan", no_wrap=True)
        table.add_column("Done", justify="right")

        # Abbreviated headers for the 10 metrics
        headers = ["Ovr", "Kg", "Ncl", "Lst", "Km", "K/K", "Cst", "Prf", "Day", "Time"]
        for h in headers:
            table.add_column(h, justify="right")

        for pol in self.policies:
            stats = self.policy_stats[pol]
            metrics = stats["metrics"]
            completed = stats["completed"]

            row = [pol[:20] + "..." if len(pol) > 20 else pol, f"{completed}/{self.n_samples}"]

            # Map SIM_METRICS values to row
            # SIM_METRICS: overflows, kg, ncol, kg_lost, km, kg/km, cost, profit, days, time
            fmts = [".1f", ".0f", ".0f", ".1f", ".0f", ".2f", ".0f", ".0f", ".0f", ".1f"]
            for key, fmt in zip(SIM_METRICS, fmts):
                stats_val = metrics.get(key, (0, 0))
                if isinstance(stats_val, (list, tuple)):
                    avg, std = stats_val
                else:
                    avg, std = stats_val, 0.0

                # Show mean ± std
                row.append(f"{avg:{fmt}}±{std:{fmt}}")

            table.add_row(*row)

        return Panel(table, title="📊 Simulation Metrics", border_style="blue")
