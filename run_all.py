"""
Unified Evaluation Runner - Wheat Segmentation CV Project
=========================================================

This script runs multiple segmentation methods (Method 1–4),
captures logs, tracks progress, and estimates remaining time.

---------------------------------------------------------
Usage:

# Run all methods (recommended)
python run_all.py

# Fast mode (for quick debugging)
python run_all.py --fast

# Run specific methods
python run_all.py --methods 1 2
python run_all.py --methods 2 4

# Skip methods
python run_all.py --skip 4

# Continue even if one method fails
python run_all.py --continue-on-error

# Dry run (only print commands)
python run_all.py --dry-run

# Specify dataset root (for Method2)
python run_all.py --dataset-root data

---------------------------------------------------------
Outputs:

output/
 ├── logs/
 │    ├── method1.log
 │    ├── method2.log
 │    ├── method3.log
 │    ├── method4.log
 │
 ├── run_summary.json

---------------------------------------------------------
Tips:

- If terminal shows no output, check logs:
    tail -f output/logs/method2.log

- Fast mode reduces:
    - Method2 image size
    - Method4 epochs

---------------------------------------------------------
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
import json


METHODS = {
    1: {"name": "Method 1 (ExG + GrabCut)", "script": "method/method1/method1.py"},
    2: {"name": "Method 2 (HSV+LBP+Gabor + SVM/RF)", "script": "method/method2/method2.py"},
    3: {"name": "Method 3 (Random Forest)", "script": "method/method3/method3.py"},
    4: {"name": "Method 4 (U-Net)", "script": "method/method4/method4.py"},
}

ESTIMATED_TIMES = {1: 20, 2: 180, 3: 60, 4: 300}


def format_time(seconds):
    if seconds is None:
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def render_progress(current, total, remaining_time):
    percent = current / total
    bar_len = 20
    filled = int(bar_len * percent)
    bar = "█" * filled + "░" * (bar_len - filled)
    return f"[{bar}] {percent*100:5.1f}% | ETA: {format_time(remaining_time)}"


def parse_args():
    parser = argparse.ArgumentParser(description="Run segmentation methods.")
    parser.add_argument("--methods", "-m", nargs="+", type=int, choices=[1,2,3,4])
    parser.add_argument("--skip", "-s", nargs="+", type=int, choices=[1,2,3,4])
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--dataset-root", type=str, default=None)
    return parser.parse_args()


def select_methods(args):
    selected = set(args.methods) if args.methods else set(METHODS.keys())
    if args.skip:
        selected -= set(args.skip)
    return [m for m in METHODS if m in selected]


def build_command(method_id, args, root):
    method = METHODS[method_id]
    script = root / method["script"]

    if not script.exists():
        raise FileNotFoundError(script)

    cmd = [sys.executable, str(script)]

    if method_id == 2:
        if args.dataset_root:
            cmd += ["--dataset-root", args.dataset_root]
        if args.fast:
            cmd += ["--image-size", "256x256"]

    if method_id == 4:
        if args.fast:
            cmd += ["--epochs", "1", "--batch-size", "2"]

    return cmd


def run_method(method_id, idx, total, args, root, log_dir):
    method = METHODS[method_id]

    print(f"\n========== [{idx}/{total}] {method['name']} ==========")
    print("→ Starting...")

    cmd = build_command(method_id, args, root)
    print("Command:", " ".join(cmd))

    if args.dry_run:
        return {"id": method_id, "status": "DRY_RUN", "time": 0}

    log_path = log_dir / f"method{method_id}.log"
    start = time.perf_counter()

    with open(log_path, "w", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        while True:
            line = process.stdout.readline()
            if not line:
                break
            print(line, end="")   
            f.write(line)

        try:
            process.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            process.kill()
            return {"id": method_id, "status": "TIMEOUT", "time": 3600}

    elapsed = time.perf_counter() - start
    status = "OK" if process.returncode == 0 else f"FAILED({process.returncode})"

    print(f"{status} in {format_time(elapsed)}")

    return {"id": method_id, "status": status, "time": elapsed}


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    log_dir = root / "output" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    selected = select_methods(args)
    total = len(selected)

    print("\n========== RUN START ==========")
    print("Running methods:", selected)

    if args.fast:
        print("⚡ Fast mode enabled")

    results = []

    for i, mid in enumerate(selected, 1):
        result = run_method(mid, i, total, args, root, log_dir)
        results.append(result)

        remaining = sum(ESTIMATED_TIMES[m] for m in selected[i:])
        progress = render_progress(i, total, remaining)

        print("\nProgress:", progress)
        print("Remaining:", format_time(remaining))

        if result["status"] != "OK" and not args.continue_on_error:
            break

    print("\n========== FINAL SUMMARY ==========")
    for r in results:
        print(f"Method {r['id']}: {r['status']} | {format_time(r['time'])}")

    summary_path = root / "output" / "run_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nSaved:", summary_path)


if __name__ == "__main__":
    main()