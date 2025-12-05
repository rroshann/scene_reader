#!/bin/bash
# Approach 2.5 Demo Launcher
# Optimized YOLO+LLM Pipeline - ~1.1s latency
# Usage: ./run_approach_2_5.sh [--mode gaming|real_world]

cd "$(dirname "$0")"
source venv/bin/activate
python demo/run_approach_2_5.py "$@"

