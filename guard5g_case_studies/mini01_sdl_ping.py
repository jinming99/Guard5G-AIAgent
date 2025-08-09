# mini01_sdl_ping.py
"""
Mini Case 01: Verify Redis/SDL connectivity if available; otherwise log and continue.
"""
import os, json
from common.util import get_logger, detect_runtime

def run():
    logger = get_logger("mini01")
    rt = detect_runtime()
    ok = False
    try:
        import redis
        r = redis.Redis(host=rt["redis_host"], port=rt["redis_port"], db=0, socket_connect_timeout=0.2)
        ok = r.ping()
    except Exception as e:
        logger.warning(f"Redis not reachable: {e}")
    logger.info(f"SDL reachable: {ok}")
    return {"title": "Mini 01 — SDL ping", "desc": "Pings Redis SDL if available.", "metrics": {}, "artifacts": [], "notes": [f"SDL reachable: {ok}"]}

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
