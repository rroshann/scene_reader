#!/bin/bash
# Approach 1.5 Demo Launcher
# Optimized Pure VLM - ~1.73s perceived latency
# Usage: ./run_approach_1_5.sh [--mode gaming|real_world]

cd "$(dirname "$0")"
source venv/bin/activate
python demo/run_approach_1_5.py "$@"

