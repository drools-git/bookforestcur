#!/usr/bin/env python3
"""BookForest2 — CLI entry point."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence noisy third-party loggers
    for lib in ["httpx", "httpcore", "chromadb", "numba", "umap"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


def cmd_run(args: argparse.Namespace) -> int:
    from src.config import load_config
    from src.pipeline import Pipeline

    cfg = load_config(args.config)
    if args.scrape:
        cfg.processing.scrape_page = True
    if args.no_cache:
        cfg.cache.enabled = False

    pipeline = Pipeline(cfg)

    if not args.skip_preflight:
        pipeline.preflight()

    bookmarks_file = Path(args.input)
    if not bookmarks_file.exists():
        logging.error("Input file not found: %s", bookmarks_file)
        return 1

    tree = pipeline.run(bookmarks_file)
    active = sum(1 for b in tree.bookmarks if not b.is_duplicate)
    dupes = sum(1 for b in tree.bookmarks if b.is_duplicate)
    print(f"\nDone. {active} bookmarks organized into {len(tree.categories)} categories. {dupes} duplicates removed.")
    print(f"Outputs written to: {cfg.output.directory}")
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    from src.config import load_config
    from src.gui.server import start_server

    cfg = load_config(args.config)
    cfg.gui.enabled = True
    if args.port:
        cfg.gui.port = args.port

    graph_json = Path(cfg.output.directory) / "graph.json"
    if not graph_json.exists():
        logging.error(
            "graph.json not found at %s — run 'main.py run' first.", graph_json
        )
        return 1

    start_server(cfg)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="bookforest2",
        description="BookForest2 — Hierarchical RAG bookmark organizer",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="PATH",
        help="Path to config.yaml (default: config.yaml)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── run ──────────────────────────────────────────────────────────────
    run_p = sub.add_parser("run", help="Process a bookmarks HTML file")
    run_p.add_argument("input", metavar="BOOKMARKS_FILE", help="Path to bookmarks.html")
    run_p.add_argument("--scrape", action="store_true", help="Enable page scraping")
    run_p.add_argument("--no-cache", action="store_true", help="Disable embedding cache")
    run_p.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip Ollama connectivity check",
    )

    # ── gui ──────────────────────────────────────────────────────────────
    gui_p = sub.add_parser("gui", help="Launch the Sigma.js visualization server")
    gui_p.add_argument("--port", type=int, default=None, help="Override GUI port")

    args = parser.parse_args()
    _setup_logging(args.verbose)

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "gui":
        return cmd_gui(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
