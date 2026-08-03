#!/usr/bin/env bash
# download_vocab.sh — Download the ORB-SLAM3 vocabulary file (ORBvoc.txt).
#
# ORB-SLAM3 requires a pre-built visual vocabulary (Bag-of-Words tree) to
# place feature descriptors into a compact representation for loop detection.
#
# The official vocabulary is hosted by the ORB-SLAM3 authors on GitHub.
# This script downloads the compressed version and extracts it to the
# standard location expected by orbslam_bridge.py.
#
# Usage:
#   cd robot/nav/orbslam3
#   bash download_vocab.sh
#
# After running, the file `robot/nav/orbslam3/ORBvoc.txt` will exist.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOCAB_FILE="$SCRIPT_DIR/ORBvoc.txt"
VOCAB_BZ2="$SCRIPT_DIR/ORBvoc.txt.bz2"

# Official ORB-SLAM3 vocabulary from the UZ-SLAMgroup GitHub release
VOCAB_URL="https://github.com/UZ-SLAMgroup/ORB_SLAM3/raw/master/Vocabulary/ORBvoc.txt.tar.gz"
VOCAB_TARBALL="$SCRIPT_DIR/ORBvoc.txt.tar.gz"

echo "[download_vocab] Checking for ORBvoc.txt in $SCRIPT_DIR …"

if [[ -f "$VOCAB_FILE" ]]; then
    SIZE=$(wc -c < "$VOCAB_FILE")
    if [[ "$SIZE" -gt 1000000 ]]; then
        echo "[download_vocab] ORBvoc.txt already present ($(du -sh "$VOCAB_FILE" | cut -f1))."
        exit 0
    else
        echo "[download_vocab] WARN: ORBvoc.txt exists but is too small ($SIZE bytes) — re-downloading."
        rm -f "$VOCAB_FILE"
    fi
fi

# ── Try the primary URL (UZ-SLAMgroup .tar.gz) ────────────────────────────────
echo "[download_vocab] Downloading from $VOCAB_URL …"
if curl -fL --progress-bar "$VOCAB_URL" -o "$VOCAB_TARBALL"; then
    echo "[download_vocab] Extracting tarball …"
    tar -xzf "$VOCAB_TARBALL" -C "$SCRIPT_DIR" --strip-components=0
    rm -f "$VOCAB_TARBALL"
    if [[ -f "$VOCAB_FILE" ]]; then
        echo "[download_vocab] ✓ ORBvoc.txt extracted successfully ($(du -sh "$VOCAB_FILE" | cut -f1))."
        exit 0
    fi
fi

# ── Fallback: bz2 from the original ORB-SLAM3 repo ───────────────────────────
VOCAB_URL_BZ2="https://github.com/raulmur/ORB_SLAM2/raw/master/Vocabulary/ORBvoc.txt.tar.gz"
echo "[download_vocab] Primary download failed. Trying fallback: $VOCAB_URL_BZ2 …"
if curl -fL --progress-bar "$VOCAB_URL_BZ2" -o "$VOCAB_TARBALL"; then
    tar -xzf "$VOCAB_TARBALL" -C "$SCRIPT_DIR" --strip-components=0
    rm -f "$VOCAB_TARBALL"
    if [[ -f "$VOCAB_FILE" ]]; then
        echo "[download_vocab] ✓ ORBvoc.txt extracted from fallback ($(du -sh "$VOCAB_FILE" | cut -f1))."
        exit 0
    fi
fi

echo "[download_vocab] ERROR: Could not download ORBvoc.txt from any source." >&2
echo "[download_vocab] Please manually place ORBvoc.txt in: $SCRIPT_DIR" >&2
echo "[download_vocab] You can find it in any ORB-SLAM3 or ORB-SLAM2 build:" >&2
echo "    ORB_SLAM3/Vocabulary/ORBvoc.txt.tar.gz  → extract here" >&2
exit 1
