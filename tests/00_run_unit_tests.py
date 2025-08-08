#!/usr/bin/env python3
import sys, os
import pytest

def main():
    # Run all unit tests under tests/unit (verbose, do not stop on fail)
    args = ["-vv", "tests/unit"]
    raise SystemExit(pytest.main(args))

if __name__ == "__main__":
    main()
