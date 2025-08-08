#!/usr/bin/env python3
import os, sys, time, logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("test.mobiflow")

# Try to import your modules
import dataset_playback as dp
import llm_control_api_client as client

def main():
    # 1) Generate & inject a tiny synthetic burst into Redis (if available)
    try:
        player = dp.DatasetPlayer(redis_host=os.getenv("REDIS_HOST","127.0.0.1"), redis_port=int(os.getenv("REDIS_PORT","6379")))
        data = player.generate_synthetic_fbs_data(duration=8)
        for r in data:
            player.inject_mobiflow_record(r)
        log.info("Injected %d synthetic MobiFlow records", len(data))
    except Exception as e:
        log.warning("Skipping Redis injection: %s", e)

    # 2) If Auditor API is up, poll once; else skip
    try:
        ok = client.wait_for_services(timeout=5, poll=1)
        if not ok: raise RuntimeError("services not ready")
        flow = client.get_mobiflow(limit=5)
        log.info("Auditor returned %d records (sample): %s", len(flow), flow[:2])
    except Exception as e:
        log.warning("Skipping API checks (Auditor/Expert not up?): %s", e)

if __name__ == "__main__":
    main()
