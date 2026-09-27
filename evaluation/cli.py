"""Explicit live execution and offline assessment reporting commands."""

import argparse
import asyncio
import json
from pathlib import Path

from app.config import Settings
from evaluation.artifacts import RUNS_DIR, atomic_json, export_results, load_run
from evaluation.runner import build_report, evaluate
from evaluation.scoring import compare_runs, score_result_file


def parser() -> argparse.ArgumentParser:
    command_line = argparse.ArgumentParser(prog="python -m evaluation.cli")
    commands = command_line.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run the real classification pipeline explicitly")
    run.add_argument("--prompt-version", required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--live", action="store_true")
    run.add_argument("--resume", action="store_true")

    score = commands.add_parser("score", help="Score existing artifacts without provider access")
    source = score.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--results", type=Path)

    export = commands.add_parser("export", help="Export a complete twelve-message run")
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--replace", action="store_true")

    compare = commands.add_parser("compare", help="Compare two completed runs offline")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path)
    return command_line


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "run":
        if not args.live:
            print("Live evaluation requires --live; no provider was called.")
            return 2
        try:
            run = asyncio.run(
                evaluate(
                    run_id=args.run_id,
                    prompt_version=args.prompt_version,
                    settings=Settings.from_env(),
                    live=True,
                    resume=args.resume,
                    runs_dir=RUNS_DIR,
                )
            )
        except Exception:
            print("Evaluation stopped. Inspect the named run artifacts if they exist.")
            return 1
        print(json.dumps(build_report(run, RUNS_DIR / args.run_id), ensure_ascii=False, indent=2))
        return 0 if all(item["state"] == "succeeded" for item in run["outcomes"]) else 1
    if args.command == "score":
        if args.run_dir is not None:
            report = build_report(load_run(args.run_dir), args.run_dir)
            atomic_json(args.run_dir / "report.json", report)
        else:
            report = {"scores": score_result_file(args.results)}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "export":
        if args.output.name != "resultados.json":
            raise ValueError("Assessment export must be named resultados.json.")
        if args.output.exists() and not args.replace:
            raise FileExistsError("Output exists; pass --replace to replace it explicitly.")
        results = export_results(load_run(args.run_dir), args.output)
        print(f"Exported {len(results)} classifications.")
        return 0
    comparison = compare_runs(load_run(args.baseline), load_run(args.candidate))
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError("Comparison output already exists.")
        atomic_json(args.output, comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
