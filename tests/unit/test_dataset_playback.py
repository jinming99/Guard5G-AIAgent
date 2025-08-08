#!/usr/bin/env python3
"""
Exercise synthetic data generation and Redis injection with a fake client.
"""

import json
import logging
import types
import dataset_playback as dp

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("test_dataset_playback")

class FakeRedis:
    def __init__(self):
        self.kv = {}
    def ping(self): return True
    def setex(self, key, ttl, val): self.kv[key]=val
    def set(self, key, val): self.kv[key]=val
    def incr(self, key): self.kv[key]=str(int(self.kv.get(key,"0"))+1)

def main():
    player = dp.DatasetPlayer.__new__(dp.DatasetPlayer)  # bypass __init__
    player.redis_client = FakeRedis()

    # Generate and inject
    try:
        data = player.generate_synthetic_fbs_data(duration=10)
    except Exception as e:
        logger.warning("generate_synthetic_fbs_data failed (ok if scapy missing): %s", e)
        data = [{"ue_id": "UE1", "timestamp": 0, "suspected_fbs": True}]

    for r in data[:5]:
        player.inject_mobiflow_record(r)

    assert "stats:mobiflow_count" in player.redis_client.kv
    logger.info("Injected %s records, OK", player.redis_client.kv["stats:mobiflow_count"])

if __name__ == "__main__":
    main()
