"""
IRIS Gate Evo — Live Terminal Dashboard

Real-time convergence metrics display. Updates in-place using ANSI escape
codes. Shows cycle, round, all four metrics, TYPE distribution, and model
status as the pipeline runs.

Usage:
    dash = Dashboard()
    dash.update_stage("S1")
    dash.update_metrics(snapshot)
    dash.update_round(3, 10)
    dash.finalize()
"""

import sys
import time
from dataclasses import dataclass
from typing import Optional


# ANSI escape codes
CLEAR_LINE = "\033[2K"
MOVE_UP = "\033[A"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
WHITE = "\033[37m"
MAGENTA = "\033[35m"


def _bar(value: float, width: int = 20, threshold: float = None) -> str:
    """Render a progress bar with optional threshold marker."""
    filled = int(value * width)
    filled = max(0, min(width, filled))

    if threshold is not None:
        thresh_pos = int(threshold * width)
        bar_chars = []
        for i in range(width):
            if i < filled:
                if value >= threshold:
                    bar_chars.append(f"{GREEN}#{RESET}")
                else:
                    bar_chars.append(f"{YELLOW}#{RESET}")
            elif i == thresh_pos:
                bar_chars.append(f"{RED}|{RESET}")
            else:
                bar_chars.append(f"{DIM}.{RESET}")
        return "".join(bar_chars)
    else:
        bar = "#" * filled + "." * (width - filled)
        return bar


class Dashboard:
    """Live terminal dashboard for IRIS pipeline metrics."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._lines_written = 0
        self._stage = ""
        self._cycle = 0
        self._round = 0
        self._max_rounds = 10
        self._total_calls = 0
        self._start_time = time.monotonic()

        # Current metrics
        self._cosine = 0.0
        self._jaccard = 0.0
        self._jsd = 0.0
        self._type01 = 0.0
        self._kappa = 0.0
        self._type_dist = {0: 0, 1: 0, 2: 0, 3: 0}
        self._n_models = 0
        self._delta = 0.0

        # Thresholds
        self._cosine_threshold = 0.85
        self._type01_threshold = 0.90
        self._jaccard_floor = 0.10

        # History for sparkline
        self._cosine_history: list[float] = []
        self._type01_history: list[float] = []

        # Gate status
        self._gate_status = ""

    def set_thresholds(self, cosine: float = 0.85, type01: float = 0.90,
                       jaccard_floor: float = 0.10):
        self._cosine_threshold = cosine
        self._type01_threshold = type01
        self._jaccard_floor = jaccard_floor

    def update_stage(self, stage: str):
        self._stage = stage

    def update_cycle(self, cycle: int):
        self._cycle = cycle
        self._cosine_history.clear()
        self._type01_history.clear()

    def update_round(self, round_num: int, max_rounds: int = 10):
        self._round = round_num
        self._max_rounds = max_rounds

    def update_calls(self, total_calls: int):
        self._total_calls = total_calls

    def update_models(self, n_models: int):
        self._n_models = n_models

    def update_metrics(self, snapshot):
        """Update from a ConvergenceSnapshot (dataclass or dict)."""
        if hasattr(snapshot, 'cosine'):
            self._cosine = snapshot.cosine
            self._jaccard = snapshot.jaccard
            self._jsd = snapshot.jsd
            self._type01 = snapshot.type_01_ratio
            self._kappa = snapshot.kappa
            self._type_dist = snapshot.type_distribution or {}
        elif isinstance(snapshot, dict):
            self._cosine = snapshot.get('cosine', 0)
            self._jaccard = snapshot.get('jaccard', 0)
            self._jsd = snapshot.get('jsd', 0)
            self._type01 = snapshot.get('type_01_ratio', 0)
            self._kappa = snapshot.get('kappa', 0)
            self._type_dist = snapshot.get('type_distribution', {})

        self._cosine_history.append(self._cosine)
        self._type01_history.append(self._type01)

    def update_delta(self, d: float):
        self._delta = d

    def set_gate_status(self, passed: bool, message: str = ""):
        if passed:
            self._gate_status = f"{GREEN}{BOLD}PASSED{RESET}"
        else:
            self._gate_status = f"{RED}{BOLD}FAILED{RESET} {DIM}{message}{RESET}"

    def _sparkline(self, values: list[float], width: int = 15) -> str:
        """Render a sparkline from recent values."""
        if not values:
            return ""
        chars = " _.-~*"
        # Take last `width` values
        recent = values[-width:]
        if max(recent) == min(recent):
            return chars[3] * len(recent)
        line = ""
        for v in recent:
            normalized = (v - min(recent)) / (max(recent) - min(recent) + 1e-9)
            idx = int(normalized * (len(chars) - 1))
            line += chars[idx]
        return line

    def render(self):
        """Render the dashboard to terminal."""
        if not self.enabled:
            return

        elapsed = time.monotonic() - self._start_time

        # Clear previous render
        if self._lines_written > 0:
            sys.stdout.write(f"\033[{self._lines_written}A")

        lines = []

        # Header
        cycle_str = f" CYCLE {self._cycle}" if self._cycle > 0 else ""
        lines.append(
            f"{BOLD}{CYAN}"
            f"{'=' * 56}"
            f"{RESET}"
        )
        lines.append(
            f"{BOLD}{CYAN}"
            f"  IRIS GATE EVO — {self._stage}{cycle_str}"
            f"  {DIM}[{elapsed:.0f}s | {self._total_calls} calls]{RESET}"
        )
        lines.append(
            f"{BOLD}{CYAN}"
            f"{'=' * 56}"
            f"{RESET}"
        )

        # Round progress
        if self._round > 0:
            pct = self._round / self._max_rounds
            bar = "#" * int(pct * 20) + "." * (20 - int(pct * 20))
            lines.append(
                f"  Round {self._round:2d}/{self._max_rounds}  "
                f"[{bar}]  "
                f"delta={self._delta:.4f}  "
                f"{self._n_models} mirrors"
            )
        else:
            lines.append(
                f"  {self._n_models} mirrors active"
            )

        lines.append("")

        # Metrics with bars and thresholds
        cos_status = GREEN if self._cosine > self._cosine_threshold else YELLOW
        cos_spark = self._sparkline(self._cosine_history)
        lines.append(
            f"  {BOLD}Cosine{RESET}   {cos_status}{self._cosine:.4f}{RESET}  "
            f"[{_bar(self._cosine, 20, self._cosine_threshold)}]  "
            f"{DIM}{cos_spark}{RESET}"
        )

        jac_status = GREEN if self._jaccard >= self._jaccard_floor else RED
        lines.append(
            f"  {BOLD}Jaccard{RESET}  {jac_status}{self._jaccard:.4f}{RESET}  "
            f"[{_bar(self._jaccard, 20, self._jaccard_floor)}]  "
            f"{DIM}floor={self._jaccard_floor}{RESET}"
        )

        type_status = GREEN if self._type01 >= self._type01_threshold else YELLOW
        type_spark = self._sparkline(self._type01_history)
        lines.append(
            f"  {BOLD}TYPE0/1{RESET}  {type_status}{self._type01:.2%}{RESET}  "
            f"[{_bar(self._type01, 20, self._type01_threshold)}]  "
            f"{DIM}{type_spark}{RESET}"
        )

        lines.append(
            f"  {BOLD}JSD{RESET}      {self._jsd:.4f}  "
            f"[{_bar(1.0 - self._jsd, 20)}]  "
            f"{DIM}kappa={self._kappa:.3f}{RESET}"
        )

        # TYPE distribution
        lines.append("")
        t0 = self._type_dist.get(0, 0)
        t1 = self._type_dist.get(1, 0)
        t2 = self._type_dist.get(2, 0)
        t3 = self._type_dist.get(3, 0)
        if isinstance(t0, float) and t0 <= 1.0:
            # Ratios
            lines.append(
                f"  TYPE  "
                f"{GREEN}T0:{t0:.0%}{RESET}  "
                f"{GREEN}T1:{t1:.0%}{RESET}  "
                f"{YELLOW}T2:{t2:.0%}{RESET}  "
                f"{RED}T3:{t3:.0%}{RESET}"
            )
        else:
            # Counts
            total = t0 + t1 + t2 + t3 or 1
            lines.append(
                f"  TYPE  "
                f"{GREEN}T0:{t0}{RESET}  "
                f"{GREEN}T1:{t1}{RESET}  "
                f"{YELLOW}T2:{t2}{RESET}  "
                f"{RED}T3:{t3}{RESET}  "
                f"{DIM}(n={total}){RESET}"
            )

        # Gate status
        if self._gate_status:
            lines.append("")
            lines.append(f"  S3 GATE: {self._gate_status}")

        lines.append(
            f"{CYAN}{'─' * 56}{RESET}"
        )

        # Write all lines
        output = "\n".join(f"{CLEAR_LINE}{line}" for line in lines)
        sys.stdout.write(output + "\n")
        sys.stdout.flush()

        self._lines_written = len(lines)

    def finalize(self):
        """Show cursor and print final newline."""
        if self.enabled:
            sys.stdout.write(SHOW_CURSOR)
            sys.stdout.flush()
