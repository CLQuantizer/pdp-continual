#!/usr/bin/env bash
# Republish https://learning.mywords.io from this repo's docs. Safe to run any time; ~1 s.
# Call after every material loop commit:  tools/publish.sh [commit-sha]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
"$PY" tools/publish_site.py --out /home/ezio/apps/learning ${1:+--commit "$1"}
