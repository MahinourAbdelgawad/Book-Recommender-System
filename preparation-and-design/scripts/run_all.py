#run all models
import subprocess, sys, time

if __package__:
    from .common import EXPERIMENT_DIR, LOGS, out_dir
    from .models import MODELS
else:
    from common import EXPERIMENT_DIR, LOGS, out_dir
    from models import MODELS

fast = "--fast" in sys.argv  # quick test before running the full thing
only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
root = out_dir(fast)

failed = []

for name in MODELS:
    if only is not None and name not in only:
        continue
    if (root / f"{name}.json").exists():
        print(f"skip {name}", flush=True)
        continue

    t0 = time.time()
    log_path = LOGS / f"{name}.log"
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "scripts.run_model", name] + (["--fast"] if fast else []),
            cwd=EXPERIMENT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        r = proc.wait()

    status = "ok" if r == 0 else f"failed({r})"
    print(f"{name}: {status} in {time.time()-t0:.0f}s", flush=True)

    if r != 0:
        failed.append(name)
        (root / f"{name}.FAILED").write_text(status)

print("\nall done. failed:", failed or "none", flush=True)

subprocess.run([sys.executable, "-m", "scripts.summarize"] + (["--fast"] if fast else []), cwd=EXPERIMENT_DIR)
