#!/usr/bin/env python3
import os, sys, time, socket

sock_path = "/tmp/oai_oran_cfg.sock"
timeout = int(os.getenv("ORAN_SOCKET_TIMEOUT", "60"))
start = time.time()

print(f"[STEP] Waiting up to {timeout}s for ORAN socket: {sock_path}")
while time.time() - start < timeout:
    if os.path.exists(sock_path):
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(1.5)
            s.connect(sock_path)
            s.sendall(b"GET_CONFIG\n")
            _ = s.recv(4096)
            s.close()
            print("[OK] ORAN socket ready")
            sys.exit(0)
        except Exception as e:
            time.sleep(1.0)
            continue
    time.sleep(1.0)

print("[WARN] ORAN socket not ready on host (OK if running via docker)")
sys.exit(0)
