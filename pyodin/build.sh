#!/usr/bin/env bash
# Build the pyodin pybind11 module and drop pyodin*.so where neer_odin can import it:
#   1) the project root (so `python -m robot.odin_pub_node` run from neer_odin/ finds it)
#   2) the active env's site-packages (so it imports from anywhere)
#
# Usage:  conda activate slam-odin && ./pyodin/build.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

if ! python -c "import pybind11" 2>/dev/null; then
  echo "[build] pybind11 not importable; installing into the active env..."
  pip install pybind11
fi
PYBIND_DIR="$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')"

BUILD_DIR="$HERE/build"
rm -rf "$BUILD_DIR" && mkdir -p "$BUILD_DIR"

cmake -S "$HERE" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$PYBIND_DIR" \
  -DPython_EXECUTABLE="$(command -v python)"

cmake --build "$BUILD_DIR" -j"$(nproc)"

SO="$(find "$BUILD_DIR" -maxdepth 1 -name 'pyodin*.so' | head -1)"
if [[ -z "$SO" ]]; then echo "[build] ERROR: pyodin*.so not produced"; exit 1; fi

cp -f "$SO" "$ROOT/"
echo "[build] copied -> $ROOT/$(basename "$SO")"

SITE="$(python - <<'PY'
import site, sys
try:
    print(site.getsitepackages()[0])
except Exception:
    print(sys.path[-1])
PY
)"
if [[ -n "${SITE:-}" && -d "$SITE" ]]; then
  cp -f "$SO" "$SITE/" && echo "[build] installed -> $SITE"
fi

python -c "import pyodin; print('[build] import OK:', [x for x in dir(pyodin.OdinSensor) if not x.startswith('_')])"
