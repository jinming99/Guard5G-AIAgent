#!/usr/bin/env python3
import os, sys, json, socket, subprocess, shlex, time

def try_host():
    path = "/tmp/oai_oran_cfg.sock"
    if not os.path.exists(path):
        return None
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(2.5)
        s.connect(path)
        s.sendall(b"GET_CONFIG\n")
        data = s.recv(65536)
        s.close()
        return json.loads(data.decode("utf-8", "ignore"))
    except Exception as e:
        print(f"[HOST] Error: {e}")
        return None

def docker_names():
    try:
        out = subprocess.check_output(["docker","ps","--format","{{.Names}}"], text=True, timeout=5)
        return {line.strip() for line in out.splitlines() if line.strip()}
    except Exception:
        return set()

def try_container(name="fbs-gnb"):
    if name not in docker_names():
        return None
    py = r'''
import socket, json, sys
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.settimeout(3.0)
s.connect("/tmp/oai_oran_cfg.sock")
s.sendall(b"GET_CONFIG\n")
d = s.recv(65536).decode("utf-8", "ignore")
print(d)
'''
    try:
        out = subprocess.check_output(["docker","exec","-i",name,"python3","-c",py], text=True, timeout=10)
        try:
            return json.loads(out)
        except Exception:
            print("[DOCKER] Raw:", out.strip())
            return None
    except Exception as e:
        print(f"[DOCKER] Error: {e}")
        return None

def main():
    print("[STEP] ORAN GET_CONFIG via host socket (if available)")
    cfg = try_host()
    if cfg:
        print("[OK] Host ORAN config keys:", list(cfg.keys())[:10])
        sys.exit(0)
    print("[INFO] Host socket not available; trying docker 'fbs-gnb'")
    cfg = try_container()
    if cfg:
        print("[OK] Docker ORAN config keys:", list(cfg.keys())[:10])
        sys.exit(0)
    print("[ERR] Could not retrieve ORAN config from host or docker")
    sys.exit(1)

if __name__ == "__main__":
    main()
