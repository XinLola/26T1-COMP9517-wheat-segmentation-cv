"""
Unified Evaluation Runner - Wheat Segmentation CV Project
=========================================================
Usage:
  python run_all.py
  python run_all.py --methods 2
  python run_all.py --methods 1 3 4
  python run_all.py --methods 2 4
  python run_all.py --skip 2
  python run_all.py --continue-on-error
  python run_all.py --dry-run
  python run_all.py --methods 2 --dataset-root data
  python run_all.py --fast
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
import json


METHODS = {
    1: {
        "name": "Method 1 (ExG + GrabCut)",
        "script": "method/method1/method1.py",
    },
    2: {
        "name": "Method 2 (HSV+LBP+Gabor + SVM/RF)",
        "script": "method/method2/method2.py",
    },
    3: {
        "name": "Method 3 (Random Forest)",
        "script": "method/method3/method3.py",
    },
    4: {
        "name": "Method 4 (U-Net)",
        "script": "method/method4/method4.py",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run and evaluate wheat segmentation methods.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--methods", "-m", nargs="+", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--skip", "-s", nargs="+", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--method2-image-size", type=str, default=None)
    parser.add_argument("--method4-epochs", type=int, default=None)
    parser.add_argument("--method4-batch-size", type=int, default=None)
    parser.add_argument("--method4-lr", type=float, default=None)
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

        if args.method2_image_size:
            cmd += ["--image-size", args.method2_image_size]
        elif args.fast:
            cmd += ["--image-size", "256x256"]

    if method_id == 4:
        if args.method4_epochs is not None:
            cmd += ["--epochs", str(args.method4_epochs)]
        elif args.fast:
            cmd += ["--epochs", "1"]

        if args.method4_batch_size is not None:
            cmd += ["--batch-size", str(args.method4_batch_size)]
        elif args.fast:
            cmd += ["--batch-size", "2"]

        if args.method4_lr is not None:
            cmd += ["--lr", str(args.method4_lr)]

    return cmd


def run_method(method_id, idx, total, args, root, log_dir):
    method = METHODS[method_id]

    print(f"\n[{idx}/{total}] {method['name']}")

    cmd = build_command(method_id, args, root)
    print("Command:", " ".join(cmd))

    if args.dry_run:
        return {
            "id": method_id,
            "status": "DRY_RUN",
            "time": 0,
            "log": None,
            "command": cmd,
        }

    log_path = log_dir / f"method{method_id}.log"
    start = time.perf_counter()

    with open(log_path, "w", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd,
            cwd=root,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        process.wait()

    elapsed = time.perf_counter() - start
    status = "OK" if process.returncode == 0 else "FAILED"

    print(f"{status} in {int(elapsed)}s | log: {log_path}")

    return {
        "id": method_id,
        "status": status,
        "time": elapsed,
        "log": str(log_path),
        "command": cmd,
    }


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    log_dir = root / "output" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    selected = select_methods(args)

    print("\nRunning methods:", selected)
    if args.fast:
        print("Fast mode: ON")

    results = []
    failed = False

    for i, mid in enumerate(selected, 1):
        try:
            result = run_method(mid, i, len(selected), args, root, log_dir)
            results.append(result)

            if result["status"] != "OK":
                failed = True
                if not args.continue_on_error:
                    break

        except Exception as e:
            print(f"ERROR: method {mid} crashed: {e}")
            failed = True
            results.append({
                "id": mid,
                "status": "CRASHED",
                "time": 0,
                "log": None,
                "command": None,
                "error": str(e),
            })
            if not args.continue_on_error:
                break

    summary_path = root / "output" / "run_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nSummary saved to:", summary_path)

    for result in results:
        print(f"[{result['id']}] {result['status']} ({int(result.get('time', 0))}s)")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()