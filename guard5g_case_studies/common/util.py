# common/util.py
import os, logging, sys, time, random

_LOGGERS = {}

def get_logger(name: str = "case_studies", logfile: str = None, level: str = None) -> logging.Logger:
    global _LOGGERS
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, (level or os.getenv("LOG_LEVEL", "INFO")).upper(), logging.INFO))
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    log_dir = os.getenv("CS_LOG_DIR", "outputs/logs")
    os.makedirs(log_dir, exist_ok=True)
    logfile = logfile or os.path.join(log_dir, f"{name}.log")
    fh = logging.FileHandler(logfile, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    _LOGGERS[name] = logger
    return logger

def set_seed(seed: int = None) -> int:
    """Set seed across random libs (Python + numpy if available)."""
    if seed is None:
        # Make a reproducible random seed if not provided, but persist for session
        seed = int(time.time()) % 10_000_000
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    return seed

def probe_url(url: str, timeout: float = 1.0) -> bool:
    """Tiny reachability probe (optional); returns False on any failure."""
    try:
        import requests
        r = requests.get(url, timeout=timeout)
        return r.status_code < 500
    except Exception:
        return False

def detect_runtime():
    """
    Best-effort detection for real stack availability vs. offline mode.
    Returns a dict with flags and derived paths.
    """
    flags = {
        "redis_host": os.getenv("REDIS_HOST", "localhost"),
        "redis_port": int(os.getenv("REDIS_PORT", "6379")),
        "auditor_api": os.getenv("AUDITOR_API", "http://localhost:8090"),
        "expert_api": os.getenv("EXPERT_API", "http://localhost:8091"),
        "oai_path": os.getenv("OAI_PATH", "/opt/oai-5g"),
        "openai_key_set": bool(os.getenv("OPENAI_API_KEY")),
    }

    flags["auditor_up"] = probe_url(flags["auditor_api"] + "/health")
    flags["expert_up"] = probe_url(flags["expert_api"] + "/health")

    # launch_fbs.sh present?
    flags["launch_fbs"] = os.path.join(flags["oai_path"], "tools", "fbs_scenarios", "launch_fbs.sh")
    flags["has_fbs_launcher"] = os.path.isfile(flags["launch_fbs"]) and os.access(flags["launch_fbs"], os.X_OK)

    # Scenario socket/file hints
    flags["oran_cfg_json"] = "/tmp/oai_oran_config.json"
    flags["oran_cfg_sock"] = "/tmp/oai_oran_cfg.sock"
    flags["oran_cfg_present"] = os.path.exists(flags["oran_cfg_json"]) or os.path.exists(flags["oran_cfg_sock"])

    # Derive offline flag
    flags["offline"] = not (flags["auditor_up"] and flags["expert_up"])  # offline if not both up
    return flags

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path
