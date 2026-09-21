#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest -q
doc-intel-factory run --n-docs 300 --seed 42
