#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# nav.sh — Launch the Odin 1 SLAM pipeline in tmux
#
# Creates a tmux session "nav" with 2 panes:
#   1. Odin publisher   (robot.odin_pub_node)   — drop-in for the old zed_pub_node
#   2. SLAM node        (robot.slam_node_)       — UNCHANGED from neer_slam
#
# Usage:
#   bash nav.sh
#   bash nav.sh --no-fresh    # keep any saved relocalization map
#   bash nav.sh --kill        # kill existing session
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SESSION="nav"
CONDA_ENV="slam-odin"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Parse args
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

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  nav — Starting Odin 1 SLAM Pipeline                          ║"
echo "║  Project: $PROJECT_DIR"
echo "║  Conda:   $CONDA_ENV"
echo "╚══════════════════════════════════════════════════════════════╝"

# Pane 1: Odin publisher
tmux new-session -d -s "$SESSION" -n "nav" \
    "echo '=== [1/2] Odin Publisher ===' && \
     source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh || true && \
     cd $PROJECT_DIR && \
     conda activate $CONDA_ENV && \
     python -m robot.odin_pub_node $FRESH_FLAG; \
     echo 'Odin publisher exited. Press Enter.'; read"

# Let the box connect + start streaming before the consumer attaches
sleep 3

# Pane 2: SLAM node (identical to neer_slam)
tmux split-window -t "$SESSION" -v \
    "echo '=== [2/2] SLAM Node ===' && \
     source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh || true && \
     cd $PROJECT_DIR && \
     conda activate $CONDA_ENV && \
     python -m robot.slam_node_; \
     echo 'SLAM node exited. Press Enter.'; read"

tmux select-layout -t "$SESSION" even-vertical

echo ""
echo "tmux session '$SESSION' started with 2 panes."
echo "Attach:  tmux attach -t $SESSION"
echo "Kill:    bash nav.sh --kill"
echo "SLAM Viser UI: http://localhost:8099"
echo ""

tmux attach -t "$SESSION"
