#!/usr/bin/env bash
# patch_no_pangolin.sh — Patches ORB-SLAM3 CMakeLists.txt to build without
# Pangolin (the GUI viewer).  Safe for headless robot deployments where
# Viewer.on=0 is set in the YAML config anyway.
#
# Usage (run from the ORB_SLAM3 root directory):
#   bash /path/to/patch_no_pangolin.sh
#
# What this does:
#   1. Removes the find_package(Pangolin REQUIRED) call
#   2. Removes Pangolin include dirs from target_include_directories
#   3. Removes Pangolin libs from target_link_libraries
#   4. Stubs out the Viewer class so no-viewer builds link cleanly
#
# The resulting binary runs ORB-SLAM3 with tracking + mapping, but no GUI.
# All pose output still works normally via the System::TrackRGBD() API.

set -euo pipefail
CMAKE="CMakeLists.txt"

if [[ ! -f "$CMAKE" ]]; then
    echo "ERROR: Run this script from the ORB_SLAM3 root directory (where CMakeLists.txt lives)."
    exit 1
fi

echo "[patch] Backing up $CMAKE → ${CMAKE}.bak"
cp "$CMAKE" "${CMAKE}.bak"

# ── 1. Comment out the Pangolin find_package ─────────────────────────────────
sed -i 's/^find_package(Pangolin REQUIRED)/# find_package(Pangolin REQUIRED) # patched: headless build/' "$CMAKE"

# ── 2. Remove Pangolin from include dirs ─────────────────────────────────────
sed -i 's/${Pangolin_INCLUDE_DIRS}//g' "$CMAKE"

# ── 3. Remove Pangolin from link libs ────────────────────────────────────────
sed -i 's/${Pangolin_LIBRARIES}//g' "$CMAKE"

# ── 4. Add NO_VIEWER compile definition so Viewer.cc is compiled as a stub ──
# Insert after the project() line so it's globally visible
sed -i '/^project(ORB_SLAM3)/a add_compile_definitions(NO_VIEWER)' "$CMAKE"

echo "[patch] CMakeLists.txt patched for headless (no-Pangolin) build."
echo ""
echo "You also need to stub the Viewer class if it references Pangolin headers."
echo "Running viewer stub patcher..."

# ── 5. Stub out Viewer.cc if it includes pangolin ────────────────────────────
VIEWER_CC="src/Viewer.cc"
if grep -q "pangolin" "$VIEWER_CC" 2>/dev/null; then
    cp "$VIEWER_CC" "${VIEWER_CC}.bak"
    cat > "$VIEWER_CC" << 'EOF'
// Viewer.cc — stubbed for headless (no-Pangolin) build
// The Viewer is disabled via Viewer.on=0 in the YAML config.
#include "Viewer.h"
namespace ORB_SLAM3 {
Viewer::Viewer(System* pSystem, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer,
               Tracking* pTracking, const string& strSettingPath, Settings* settings)
{}
void Viewer::Run() {}
void Viewer::RequestFinish() {}
void Viewer::RequestStop() {}
bool Viewer::isFinished() { return true; }
bool Viewer::isStopped() { return true; }
bool Viewer::Stop() { return true; }
void Viewer::Release() {}
void Viewer::SetCurrentCameraPose(const Sophus::SE3f&) {}
bool Viewer::both_stopped() { return true; }
}
EOF
    echo "[patch] Viewer.cc stubbed (original saved as Viewer.cc.bak)"
else
    echo "[patch] Viewer.cc has no pangolin includes — no stub needed."
fi

echo ""
echo "[patch] Done. Now run: bash build.sh"
