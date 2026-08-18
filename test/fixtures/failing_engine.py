#!/usr/bin/env python3
import sys

print("deliberate engine failure", file=sys.stderr)
raise SystemExit(7)
