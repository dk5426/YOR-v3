#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# nav.sh — launch the SLAM pipeline in tmux.
#
# Two panes:
#   1. robot/odin_pub_node.py   Odin 1 sensor publisher → slam/* on :6000
#   2. robot/slam_node_.py      voxel map + A* + Viser UI on :8099
#
# The robot side (robot/yor.py) is NOT started here — run it separately on the
# Pi. Without it slam_node_ still maps and plans; only follow_path and the EKF's
# encoder predict step go quiet.
#
#   bash nav.sh                # fresh map
#   bash nav.sh --no-fresh     # reuse a saved relocalization map
#   bash nav.sh --kill         # stop
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SESSION="nav"
CONDA_ENV="${YOR_CONDA_ENV:-slam-odin}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

FRESH_FLAG="--fresh"
for arg in "$@"; do
    case "$arg" in
        --no-fresh) FRESH_FLAG="" ;;
        --kill)
            tmux kill-session -t "$SESSION" 2>/dev/null && echo "Killed session $SESSION" || echo "No session $SESSION"
            exit 0
            ;;
    esac
done

tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "nav — starting SLAM pipeline"
echo "  project: $PROJECT_DIR"
echo "  conda:   $CONDA_ENV"

CONDA_SETUP="source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh || true"

# Pane 1: Odin publisher
tmux new-session -d -s "$SESSION" -n "nav" \
    "echo '=== [1/2] Odin publisher ===' && \
     $CONDA_SETUP && \
     cd $PROJECT_DIR && \
     conda activate $CONDA_ENV && \
     python -m robot.odin_pub_node $FRESH_FLAG; \
     echo 'Odin publisher exited. Press Enter.'; read"

# Let the box connect and start streaming before the consumer attaches.
sleep 3

# Pane 2: SLAM node
tmux split-window -t "$SESSION" -v \
    "echo '=== [2/2] SLAM node ===' && \
     $CONDA_SETUP && \
     cd $PROJECT_DIR && \
     conda activate $CONDA_ENV && \
     python -m robot.slam_node_; \
     echo 'SLAM node exited. Press Enter.'; read"

tmux select-layout -t "$SESSION" even-vertical

echo
echo "tmux session '$SESSION' started."
echo "  attach: tmux attach -t $SESSION"
echo "  kill:   bash nav.sh --kill"
echo "  UI:     http://<robot-ip>:8099"
echo

tmux attach -t "$SESSION"
