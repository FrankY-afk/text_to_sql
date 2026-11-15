"""Main entrypoint for Text-to-SQL (Spider 2.0 + MCP).

Integrates Model Context Protocol (MCP), Adaptive CTE Planning,
Semantic Validation, and Execution-Grounded Self-Repair.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_demo import run_demonstration
from src.cli import launch_cli


def main():
    parser = argparse.ArgumentParser(
        description="Text-to-SQL Assistant: Spider 2.0 + MCP + Adaptive CTE"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the automated demonstration runner",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Launch the interactive terminal console (default)",
    )
    args = parser.parse_args()

    if args.demo:
        run_demonstration()
    else:
        # Default behavior: Launch interactive terminal console
        launch_cli()


if __name__ == "__main__":
    main()
