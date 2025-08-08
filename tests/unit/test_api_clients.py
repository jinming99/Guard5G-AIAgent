#!/usr/bin/env python3
"""
Tests for llm_control_api_client.py: health checks, waits, realtime monitor
"""

import json
import time
import types
import logging
from unittest.mock import patch, MagicMock

import llm_control_api_client as api

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("test_api_clients")

class DummyResp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload or {"status":"healthy"}
        self.content = json.dumps(self._payload).encode()

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"http {self.status_code}")

    def json(self):
        return self._payload

def main():
    # Patch requests.Session.request to return healthy response
    with patch.object(api.APIClient, "_request") as req:
        req.return_value = {"status": "ok"}
        assert api.APIClient("http://x").health_check() is True
        logger.info("health_check ok")

    # Test wait_for_services() loop behavior
    with patch.object(api, "auditor") as auditor, patch.object(api, "expert") as expert:
        auditor.health_check.return_value = False
        expert.health_check.return_value  = True
        t0 = time.time()
        ok = api.wait_for_services(timeout=2, poll=0.5)
        logger.info("wait_for_services returned: %s (%.2fs)", ok, time.time()-t0)

    # Test monitor_realtime dedup: feed repeating records
    records = [
        {"ue_id":"A", "timestamp":1, "suspected_fbs":False},
        {"ue_id":"B", "timestamp":2, "suspected_fbs":True, "msg_id":42},
        {"ue_id":"B", "timestamp":2, "suspected_fbs":True, "msg_id":42},  # duplicate
    ]
    seen = []
    def cb(new):
        logger.info("callback got %d records: %s", len(new), new)
        seen.extend(new)

    with patch.object(api, "get_mobiflow") as gm:
        gm.side_effect = [records, records, [], []]
        api.monitor_realtime(callback=cb, interval=0, duration=0.2)

    assert len(seen) == 2, f"expected 2 unique, got {len(seen)}"
    logger.info("monitor_realtime dedup OK")

if __name__ == "__main__":
    main()
