#!/usr/bin/env python3
import argparse, os, sys, subprocess, time, yaml, datetime, pathlib, shlex

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent

def nowstamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def load_manifest(path: pathlib.Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def detect_cmd(fpath: pathlib.Path):
    if fpath.suffix == ".py":
        return [sys.executable, "-u", str(fpath)]
    if fpath.suffix == ".sh":
        return ["bash", str(fpath)]
    return [sys.executable, "-u", str(fpath)]

def stream_process(proc, timeout, test_name):
    start = time.time()
    lines = []
    while True:
        if timeout and (time.time() - start) > timeout:
            try:
                proc.kill()
            except Exception:
                pass
            lines.append(f"[TIMEOUT] {test_name} exceeded {timeout}s\n")
            break
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            lines.append(line)
    rc = proc.returncode if proc.returncode is not None else 124
    return rc, lines

def run_one(test_path: pathlib.Path, timeout: int, log_dir: pathlib.Path):
    test_name = test_path.name
    cmd = detect_cmd(test_path)
    env = os.environ.copy()
    per_test_log = log_dir / f"{test_name}.{nowstamp()}.log"
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=REPO,
        env=env,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    rc, lines = stream_process(proc, timeout, test_name)
    per_test_log.write_text("".join(lines), encoding="utf-8")
    return rc, per_test_log

def main():
    ap = argparse.ArgumentParser(description="Run staged tests and capture logs")
    ap.add_argument("--stage", choices=["unit","env","simulator","oran","rules","agent","e2e","all"], default="all",
                    help="Which stage to run")
    ap.add_argument("--manifest", default=str(HERE / "stages.yaml"), help="Path to stages.yaml")
    ap.add_argument("--timeout", type=int, default=600, help="Per-test timeout in seconds")
    ap.add_argument("--output", default=None, help="Aggregate log file path")
    ap.add_argument("--stop-on-fail", action="store_true", help="Stop on first failure")
    ap.add_argument("--tail", type=int, default=120, help="Lines of each test log to include in aggregate (on PASS)")
    ap.add_argument("--fail-tail", type=int, default=400, help="Lines of each test log to include in aggregate (on FAIL)")
    args = ap.parse_args()

    manifest = load_manifest(pathlib.Path(args.manifest))
    stages = manifest.get("stages", {})
    stage_order = ["unit","env","simulator","oran","rules","agent","e2e"]
    to_run = stage_order if args.stage == "all" else [args.stage]

    logs_dir = HERE / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    agg_log = pathlib.Path(args.output) if args.output else logs_dir / f"stage-{args.stage}-{nowstamp()}.log"

    summary = []
    with agg_log.open("w", encoding="utf-8") as outf:
        outf.write(f"# STAGED TEST RUN — stage={args.stage} at {nowstamp()}\n")
        outf.write(f"# repo={REPO}\n\n")
        outf.write("# HINT: re-run with --tail N or --fail-tail M to control snippet length in this aggregate log.\n")

        for stage in to_run:
            tests = stages.get(stage, [])
            outf.write(f"\n=== STAGE: {stage} ===\n")
            if not tests:
                outf.write(f"[SKIP] No tests listed for stage '{stage}'\n")
                continue
            for test in tests:
                tpath = REPO / test
                if not tpath.exists():
                    outf.write(f"[SKIP] {test} (not found)\n")
                    continue
                outf.write(f"\n--- RUN: {test} ---\n")
                outf.flush()
                rc, perlog = run_one(tpath, timeout=args.timeout, log_dir=logs_dir)
                try:
                    content = perlog.read_text(encoding="utf-8").splitlines()
                except Exception:
                    content = ["<failed to read per-test log>"]
                status = "PASS" if rc == 0 else f"FAIL({rc})"
                tail_n = args.tail if status.startswith("PASS") else args.fail_tail
                tail = "\n".join(content[-tail_n:])
                outf.write(tail + "\n")
                summary.append((stage, test, status, str(perlog)))
                outf.write(f"--- END: {test} -> {status}; full log: {perlog}\n")
                outf.flush()
                if rc != 0 and args.stop_on_fail:
                    outf.write(f"\n[STOP-ON-FAIL] Halting after failure in {test}\n")
                    break

        outf.write("\n=== SUMMARY ===\n")
        for (st, t, s, pl) in summary:
            outf.write(f"{s:>8} | {st:<10} | {t} | {pl}\n")

    failed = [x for x in summary if not x[2].startswith("PASS")]
    if failed:
        print(f"[DONE] Some tests failed. See: {agg_log}")
        sys.exit(1)
    print(f"[DONE] All tests passed. See: {agg_log}")

if __name__ == "__main__":
    main()
