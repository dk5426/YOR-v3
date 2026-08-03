#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# orbv2/run.sh — Launch the full orbv2 ORB-SLAM3 pipeline in tmux
#
# Creates a tmux session "orbv2" with 3 panes:
#   1. ZED publisher         (zed_pub_node --fresh)
#   2. ORB-SLAM3 bridge      (orbv2.orb_bridge)
#   3. SLAM node with ORB    (orbv2.orb_slam_node)
#
# Usage:
#   bash orbv2/run.sh
#   bash orbv2/run.sh --no-fresh    # don't delete saved_map.area
#   bash orbv2/run.sh --kill        # kill existing session
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SESSION="orbv2"
CONDA_ENV="slam-zed"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Parse args
FRESH_FLAG="--fresh"
for arg in "$@"; do
    case "$arg" in
        --no-fresh) FRESH_FLAG="" ;;
        --kill)
            echo -n "Sending graceful shutdown to save map... "
            pkill -INT -f "python -m orbv2.orb_slam_node" || true
            while ps -C python,python3 -o args= 2>/dev/null | grep -q "orbv2.orb_slam_node"; do
                echo -n "."
                sleep 1
            done
            echo " Done!"
            tmux kill-session -t "$SESSION" 2>/dev/null && echo "Killed session $SESSION" || echo "No session $SESSION"
            exit 0
            ;;
    esac
done

# Kill existing session if any gracefully
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo -n "Sending graceful shutdown to previous session to save map... "
    pkill -INT -f "python -m orbv2.orb_slam_node" || true
    while ps -C python,python3 -o args= 2>/dev/null | grep -q "orbv2.orb_slam_node"; do
        echo -n "."
        sleep 1
    done
    echo " Done!"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
fi

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  orbv2 — Starting ORB-SLAM3 Loop-Closure Pipeline          ║"
echo "║  Project: $PROJECT_DIR"
echo "║  Conda:   $CONDA_ENV"
echo "╚══════════════════════════════════════════════════════════════╝"

# Create tmux session with first pane: ZED publisher
tmux new-session -d -s "$SESSION" -n "orbv2" \
    "echo '=== [1/3] ZED Publisher ===' && \
     source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh || true && \
     cd $PROJECT_DIR && \
     conda activate $CONDA_ENV && \
     python -m robot.zed_pub_node $FRESH_FLAG; \
     echo 'ZED publisher exited. Press Enter.'; read"

# Wait a moment for ZED to initialize
sleep 1

# Split vertically: ORB bridge
tmux split-window -t "$SESSION" -v \
    "echo '=== [2/3] ORB-SLAM3 Bridge ===' && \
     echo 'Waiting 5s for ZED to initialize...' && \
     sleep 5 && \
     source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh || true && \
     cd $PROJECT_DIR && \
     conda activate $CONDA_ENV && \
     python -m orbv2.orb_bridge --gen-config; \
     echo 'ORB bridge exited. Press Enter.'; read"

# Split again: SLAM node
tmux split-window -t "$SESSION" -v \
    "echo '=== [3/3] orbv2 SLAM Node ===' && \
     echo 'Waiting 30s for ORB-SLAM3 to initialize...' && \
     sleep 30 && \
     source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh || true && \
     cd $PROJECT_DIR && \
     conda activate $CONDA_ENV && \
     python -m orbv2.orb_slam_node --predict-hz 5 --save; \
     echo 'SLAM node exited. Press Enter.'; read"

# Balance panes
tmux select-layout -t "$SESSION" even-vertical

echo ""
echo "tmux session '$SESSION' started with 3 panes."
echo "Attach with:   tmux attach -t $SESSION"
echo "Kill with:     bash orbv2/run.sh --kill"
echo ""

# Attach to the session
tmux attach -t "$SESSION"
