
---

## 11) New staged runner toggles & behavior (updated)

Use the staged runner to execute tests **in dependency order** and capture logs:

```bash
python tests/run_staged_tests.py --stage all             # run everything
python tests/run_staged_tests.py --stage unit            # unit tests only
python tests/run_staged_tests.py --stage simulator       # simulator checks
python tests/run_staged_tests.py --stage e2e --timeout 1200 --tail 120 --fail-tail 600
```

**Toggles (env vars):**
- `USE_COMPOSE=1` — prefer Docker Compose services (auto-detected if `llm_fbs_utils/docker-compose.yml` exists).
- `STAGED_AUTOSTART_SERVICES=1` — auto-start Redis/Auditor/Expert and the gNB (host via `launch_fbs.sh` if available, else compose).
- `COMPOSE_FILE=llm_fbs_utils/docker-compose.yml` — override compose file path.
- `LAUNCH=/path/to/OAI-5G/tools/fbs_scenarios/launch_fbs.sh` — host-run gNB launcher.
- `ORAN_SOCKET_TIMEOUT=60` — wait budget (seconds) for `/tmp/oai_oran_cfg.sock` readiness.

**Aggregate log sizing:**
- `--tail N` controls how many lines from each test are included on PASS (default: 120).
- `--fail-tail M` controls the tail on FAIL (default: 400). Per-test **full** logs are always in `tests/logs/`.

**Stages (expanded):**
1. `unit` → runs `pytest -vv tests/unit`
2. `env` → checks Python deps and optionally starts compose services and/or host `launch_fbs.sh`
3. `simulator` → verifies scenario banner (host), basic timing waits
4. `oran` → fetches ORAN config via host socket or inside docker (`fbs-gnb`)
5. `rules` → hot‑patch cycle sanity (valid rule accepted, invalid rejected)
6. `agent` → offline workflow smoke (`LLMOrchestrator` single-round)
7. `e2e` → synthetic pipeline + your comprehensive tests under `llm_fbs_utils/tests/`

**No manual steps required:** When `STAGED_AUTOSTART_SERVICES=1` is set (default), the runner will **start what it can**:
- If `LAUNCH` exists, it will start the host fake gNB with a small scenario JSON.
- Otherwise, if Docker Compose is present, it will start `redis-sdl`, `mobiflow-auditor`, `mobiexpert`, and `oai-gnb` as defined in your compose file.

> Host vs Docker: the `oran` stage will try `/tmp/oai_oran_cfg.sock` on the host first; if not found, it will **exec into the `fbs-gnb` container** to run a Python one‑liner that queries the UNIX socket from inside the container.

---

## 12) Venv vs Docker for development (updated)

- **Fast, Python-only dev** → `./scripts/setup_venv.sh --dev` (minimal deps).
- **Comprehensive dev** → `./scripts/setup_venv.sh --full` (installs everything from `llm_fbs_utils/requirements.txt`, *which already includes pytest/black/flake8 etc.*).
- **Reproducible stack** → `docker compose -f llm_fbs_utils/docker-compose.yml up -d` (full services).

After the venv is ready, you can immediately run:

```bash
python tests/run_staged_tests.py --stage unit
python tests/run_staged_tests.py --stage env
python tests/run_staged_tests.py --stage all
```

If you prefer **compose** for simulator/services, set:

```bash
export USE_COMPOSE=1
python tests/run_staged_tests.py --stage all
```

> For host-based simulator runs, set `LAUNCH` to your `launch_fbs.sh` path. The runner will create a small scenario JSON and start it for you.

